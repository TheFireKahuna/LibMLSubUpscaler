import CoreML
import CoreVideo
import Foundation
import Metal

/// ArtCNN on the Neural Engine through Core ML (``ArtCNNCoreMLModels``), for ``SubtitleUpscaler``. The mask, the
/// window gather and the composite are GPU passes; the network runs on the Neural Engine, which leaves the GPU to the
/// video while a subtitle upscales.
///
/// The models have fixed window shapes (``SubtitleUpscaleTilePlan`` covers the kept area with them), each window a
/// core plus the network's 7-pixel reach, transparent beyond the picture, so every core pixel is the network's exact
/// output for the picture on its transparent canvas, as on the GPU. Each window's planes are IOSurface-backed
/// half-float pixel buffers (``Surfaces``) that Metal and Core ML both wrap, so the GPU writes what the Neural Engine
/// reads and reads what it writes, and nothing is copied between them.
enum SubtitleUpscaleNeuralEngine {
    struct Shape: Hashable {
        var height: Int
        var width: Int
        var function: String { "h\(height)w\(width)" }
    }

    static let shapes = ArtCNNCoreMLModels.windowHeights.flatMap { h in ArtCNNCoreMLModels.windowWidths.map { Shape(height: h, width: $0) } }

    /// The networks on a window's two planes.
    struct Pair: Hashable {
        var alpha: ArtCNNModel
        var luma: ArtCNNModel
        var name: String { "\(alpha.rawValue)_\(luma.rawValue)" }
    }

    private struct Key: Hashable {
        var pair: Pair
        var shape: Shape
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var models: [Key: Result<MLModel, Error>] = [:]

    /// The pair's function for `shape` from the package's compiled model, loaded for the CPU and Neural Engine and
    /// specialised for prediction latency. Blocks for the load, about 150 ms for each function the first time in a
    /// process; ``prewarm(_:)`` does them all in the background.
    static func model(_ pair: Pair, _ shape: Shape) throws -> MLModel {
        lock.lock()
        defer { lock.unlock() }
        let key = Key(pair: pair, shape: shape)
        if let cached = models[key] { return try cached.get() }
        let made = Result { try load(pair, shape) }
        models[key] = made
        return try made.get()
    }

    /// Where an app-open prewarm's time went, in wall milliseconds.
    struct PrewarmTimes {
        /// Loading every function.
        var load = 0.0
        /// Every function's first prediction, which has the Neural Engine compile its program.
        var firstPrediction = 0.0
        var functions = 0
        var total: Double { load + firstPrediction }
    }

    /// Loads every function of `pair` and runs each once, so nothing is left for the first subtitle but the
    /// prediction itself.
    @discardableResult
    static func prewarm(_ pair: Pair) throws -> PrewarmTimes {
        var times = PrewarmTimes()
        func lap(_ t0: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 }
        for shape in shapes {
            var t0 = DispatchTime.now().uptimeNanoseconds
            let loaded = try model(pair, shape)
            times.load += lap(t0)
            let dims = [1, 1, shape.height, shape.width].map { NSNumber(value: $0) }
            let features = try MLDictionaryFeatureProvider(dictionary: [
                "xa": MLFeatureValue(multiArray: try MLMultiArray(shape: dims, dataType: .float16)),
                "xl": MLFeatureValue(multiArray: try MLMultiArray(shape: dims, dataType: .float16)),
            ])
            t0 = DispatchTime.now().uptimeNanoseconds
            _ = try loaded.prediction(from: features)
            times.firstPrediction += lap(t0)
            times.functions += 1
        }
        return times
    }

    /// Drops the loaded functions, as a new process starts (the system keeps its Neural Engine compilations). For the
    /// benchmark.
    static func unloadModels() {
        lock.withLock { models.removeAll() }
    }

    private static func load(_ pair: Pair, _ shape: Shape) throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        configuration.functionName = shape.function
        var hints = MLOptimizationHints()
        hints.specializationStrategy = .fastPrediction
        configuration.optimizationHints = hints
        return try MLModel(contentsOf: try ArtCNNCoreMLModels.url(alpha: pair.alpha, luma: pair.luma), configuration: configuration)
    }

    /// Runs `model` on one window's two planes (``Surfaces``), one call for both. Core ML writes the outputs into
    /// their backings; an output anywhere else is an error, not a copy.
    static func predict(_ model: MLModel, _ surfaces: Surfaces) throws {
        let result = try model.prediction(from: surfaces.inputs, options: surfaces.options)
        for (name, plane) in [("ya", surfaces.alphaOut), ("yl", surfaces.lumaOut)] {
            guard let written = result.featureValue(for: name)?.multiArrayValue?.pixelBuffer, written === plane.pixelBuffer else {
                throw SubtitleUpscaler.UpscaleError.unavailable("Core ML did not write the \(name) backing")
            }
        }
    }

    /// The window planes of one shape: alpha and premultiplied luma in, their four sub-pixel channels out, each an
    /// IOSurface-backed `OneComponent16Half` pixel buffer wrapped once as an `r16Float` texture and once as a Core ML
    /// array. The input is `(1, 1, height, width)` over `height` rows; the output `(1, 4, height, width)` over
    /// `4 * height` rows, channel k from row `k * height`. IOSurface rows are padded to the Neural Engine's
    /// alignment.
    final class Surfaces {
        struct Plane {
            let pixelBuffer: CVPixelBuffer
            let texture: MTLTexture
            let array: MLMultiArray
        }

        let shape: Shape
        let alphaIn: Plane
        let lumaIn: Plane
        let alphaOut: Plane
        let lumaOut: Plane
        /// The prediction's inputs and output backings, made once: every call on these surfaces is the same.
        let inputs: MLFeatureProvider
        let options: MLPredictionOptions

        init(shape: Shape, device: MTLDevice) throws {
            self.shape = shape
            let h = shape.height, w = shape.width
            alphaIn = try Self.plane(device, width: w, rows: h, shape: [1, 1, h, w])
            lumaIn = try Self.plane(device, width: w, rows: h, shape: [1, 1, h, w])
            alphaOut = try Self.plane(device, width: w, rows: 4 * h, shape: [1, 4, h, w])
            lumaOut = try Self.plane(device, width: w, rows: 4 * h, shape: [1, 4, h, w])
            inputs = try MLDictionaryFeatureProvider(dictionary: ["xa": MLFeatureValue(multiArray: alphaIn.array),
                                                                  "xl": MLFeatureValue(multiArray: lumaIn.array)])
            options = MLPredictionOptions()
            options.outputBackings = ["ya": alphaOut.array, "yl": lumaOut.array]
        }

        private static func plane(_ device: MTLDevice, width: Int, rows: Int, shape: [Int]) throws -> Plane {
            var made: CVPixelBuffer?
            let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any],
                                               kCVPixelBufferMetalCompatibilityKey: true]
            guard CVPixelBufferCreate(nil, width, rows, kCVPixelFormatType_OneComponent16Half, attributes as CFDictionary, &made) == kCVReturnSuccess,
                  let pixelBuffer = made, let surface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() else {
                throw SubtitleUpscaler.UpscaleError.unavailable("no \(width)x\(rows) half-float surface")
            }
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: width, height: rows, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .shared
            guard let texture = device.makeTexture(descriptor: d, iosurface: surface, plane: 0) else {
                throw SubtitleUpscaler.UpscaleError.unavailable("no texture over a \(width)x\(rows) surface")
            }
            return Plane(pixelBuffer: pixelBuffer, texture: texture, array: MLMultiArray(pixelBuffer: pixelBuffer, shape: shape.map { NSNumber(value: $0) }))
        }

        /// The surfaces a plan's windows use, in window order: the k-th window of a shape takes that shape's k-th
        /// surfaces, made on first use and kept.
        final class Pool {
            let device: MTLDevice
            private var byShape: [Shape: [Surfaces]] = [:]

            init(device: MTLDevice) { self.device = device }

            func surfaces(for windows: [SubtitleUpscaleTilePlan.Window]) throws -> [Surfaces] {
                var used: [Shape: Int] = [:]
                return try windows.map { w in
                    let shape = Shape(height: w.height, width: w.width)
                    let k = used[shape, default: 0]
                    used[shape] = k + 1
                    if byShape[shape, default: []].count == k { byShape[shape, default: []].append(try Surfaces(shape: shape, device: device)) }
                    return byShape[shape]![k]
                }
            }
        }
    }
}
