import CoreML
import Foundation
import Metal

extension SubtitleUpscaler {

    /// A benchmark picture: a cropped strip or a full canvas of subtitle text.
    struct BenchmarkCase {
        let name: String
        let picture: SubtitleUpscaleFixtures.Picture
    }

    static func benchmarkCases() -> [BenchmarkCase] {
        let two = ["I never said she stole the money, and you know it.", "Then who did? Somebody in this house did."]
        return [
            BenchmarkCase(name: "1400x120 strip", picture: SubtitleUpscaleFixtures.glyphs(width: 1400, height: 120, lines: two, fontSize: 44)),
            BenchmarkCase(name: "1920x200 strip", picture: SubtitleUpscaleFixtures.glyphs(width: 1920, height: 200, lines: two, fontSize: 64)),
            BenchmarkCase(name: "1920x1080 canvas", picture: SubtitleUpscaleFixtures.glyphs(width: 1920, height: 1080, lines: two, fontSize: 56,
                                                                                            baselineFromBottom: 70)),
            // As a PGS object is cropped: two centred lines of different lengths, and one line.
            BenchmarkCase(name: "1500x150 uneven lines", picture: SubtitleUpscaleFixtures.glyphs(width: 1500, height: 150,
                                                                                                 lines: [two[0], "Then who did?"], fontSize: 56)),
            BenchmarkCase(name: "900x80 one line", picture: SubtitleUpscaleFixtures.glyphs(width: 900, height: 80, lines: ["Then who did?"], fontSize: 56)),
        ]
    }

    /// Percentiles of a set of samples, nearest rank.
    struct Spread {
        var p50 = 0.0
        var p95 = 0.0
        var p99 = 0.0
        var max = 0.0

        init(_ samples: [Double]) {
            guard !samples.isEmpty else { return }
            let s = samples.sorted()
            func rank(_ p: Double) -> Double { s[Swift.min(s.count - 1, Int((p * Double(s.count)).rounded(.up)) - 1)] }
            p50 = rank(0.5); p95 = rank(0.95); p99 = rank(0.99); max = s.last!
        }

        var text: String { String(format: "p50 %6.2f  p95 %6.2f  p99 %6.2f  max %6.2f", p50, p95, p99, max) }
    }

    /// The upscaler's costs on this device, in wall milliseconds unless marked, for a probe app to call (it blocks
    /// for some seconds). Three phases kept apart, because only the last is paid per subtitle:
    ///
    /// 1. App open: what ``prewarm(_:)`` does, timed as it does it. The GPU pipelines' compile; then the
    ///    Neural Engine models first in this process (cold when the system has not compiled them for the Neural
    ///    Engine since the install, an OS update or a purged cache; nothing here can purge that, `coldLoadReport()`
    ///    measures cold loads) and again after unloading them (warm, as on a later launch), each split into the
    ///    function loads and the first predictions; then the `fastPrediction` replacements behind them.
    /// 2. A new upscaler's ``prepare()`` (its surfaces and their first use), and the first picture on it, against
    ///    the first picture on an upscaler that was not prepared.
    /// 3. Steady state: `iterations` pictures on that upscaler, as percentiles, in all and per stage (the plan on the
    ///    CPU, GPU gather, predictions, GPU composite).
    ///
    /// Then each window shape's prediction time, which ``SubtitleUpscaleTilePlan``'s estimates come from. The GPU
    /// passes go through `submission`.
    public static func benchmark(iterations: Int = 40, submission: any SubtitleUpscaleSubmission) -> String {
        func ms(_ t0: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 }
        var lines: [String] = []
        do {
            // 1. App open.
            var t0 = DispatchTime.now().uptimeNanoseconds
            let fresh = try SubtitleUpscaleKernels(submission: submission)
            let kernelCompile = ms(t0)
            lines.append("SubtitleUpscaler benchmark on \(fresh.device.name), wall ms")
            lines.append("1. app open (prewarm)")
            lines.append(String(format: "  GPU pipelines: %.1f", kernelCompile))
            let first = try SubtitleUpscaleNeuralEngine.prewarm(Configuration.balanced.pair, upgrading: false)
            SubtitleUpscaleNeuralEngine.unloadModels()
            let again = try SubtitleUpscaleNeuralEngine.prewarm(Configuration.balanced.pair, upgrading: false)
            for (name, t) in [("first in process", first), ("reloaded", again)] {
                lines.append(String(format: "  Neural Engine, balanced %@: total %.1f = %d function loads %.1f + first predictions %.1f",
                                    name as NSString, t.total, t.functions, t.load, t.firstPrediction))
            }
            let upgrade = try SubtitleUpscaleNeuralEngine.upgrade(Configuration.balanced.pair)
            lines.append(String(format: "  then the fastPrediction functions behind them: %.1f", upgrade))
            for c in [Configuration.c4f16, .c4f32] {
                try SubtitleUpscaleNeuralEngine.prewarm(c.pair, upgrading: false)
                try SubtitleUpscaleNeuralEngine.upgrade(c.pair)
            }

            // 2 and 3. Per picture, after the prewarm.
            _ = try SubtitleUpscaleKernels.get(submission)
            lines.append("2. prepare, first picture (unprepared upscaler's first) | 3. steady state over \(iterations) pictures")
            let configurations: [(String, Configuration)] = [("C4F16", .c4f16), ("C4F32", .c4f32), ("balanced", .balanced)]
            for c in benchmarkCases() {
                for (name, configuration) in configurations {
                    let cold = try SubtitleUpscaler(configuration: configuration, submission: submission).measureRun(c.picture, iterations: 0)
                    let upscaler = try SubtitleUpscaler(configuration: configuration, submission: submission)
                    t0 = DispatchTime.now().uptimeNanoseconds
                    try upscaler.prepare()
                    let prepare = ms(t0)
                    let run = try upscaler.measureRun(c.picture, iterations: iterations)
                    lines.append(String(format: "  %@ %-8@ prepare %6.1f, first %6.2f (%6.2f) | %@", c.name as NSString, name as NSString,
                                        prepare, run.first, cold.first, run.wall.text))
                    lines.append("      plan      \(run.plan.text)")
                    lines.append("      gather    \(run.gather.text)")
                    lines.append("      predict   \(run.predict.text)   (\(run.windows) windows, 2 planes)")
                    lines.append("      composite \(run.composite.text)")
                }
            }

            lines.append("4. one prediction (both planes) per window shape, p50 (ns per window pixel and plane)")
            for alpha in ArtCNNModel.allCases {
                for luma in ArtCNNModel.allCases {
                    lines.append(try SubtitleUpscaleNeuralEngine.calibrationLine(.init(alpha: alpha, luma: luma), iterations: iterations / 2))
                }
            }

            return lines.joined(separator: "\n")
        } catch {
            lines.append("SubtitleUpscaler benchmark failed: \(error)")
            return lines.joined(separator: "\n")
        }
    }

    /// One picture through ``upscale(_:)`` (bounded, as a caller with its display set's object rectangles passes
    /// them): the first call on this upscaler, then `iterations` more, wall time and stages.
    func measureRun(_ picture: SubtitleUpscaleFixtures.Picture, iterations: Int)
        throws -> (first: Double, wall: Spread, plan: Spread, gather: Spread, predict: Spread, composite: Spread, windows: Int) {
        guard let source = SubtitleUpscaleFixtures.texture(picture, device: device) else { throw UpscaleError.unavailable("no texture") }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 2 * picture.width, height: 2 * picture.height,
                                                         mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        guard let destination = device.makeTexture(descriptor: d) else { throw UpscaleError.unavailable("no texture") }
        let content = picture.rgba.withUnsafeBytes {
            Content.scan($0.baseAddress!, width: picture.width, height: picture.height, bytesPerRow: picture.width * 4, alphaOffset: 3) ?? .empty
        }
        let job = Job(source: source, destination: destination, content: content)
        var t0 = DispatchTime.now().uptimeNanoseconds
        try upscale([job])
        let first = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        var wall: [Double] = [], plan: [Double] = [], gather: [Double] = [], predict: [Double] = [], composite: [Double] = []
        for _ in 0..<iterations {
            t0 = DispatchTime.now().uptimeNanoseconds
            try upscale([job])
            wall.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
            let s = lastNeuralEngineStages
            plan.append(s.plan); gather.append(s.gather); predict.append(s.predict); composite.append(s.composite)
        }
        return (first, Spread(wall), Spread(plan), Spread(gather), Spread(predict), Spread(composite), lastNeuralEngineStages.tiles)
    }
}

extension SubtitleUpscaleNeuralEngine {
    /// Each window shape's synchronous prediction time for `pair` on its own surfaces, median, with the time per
    /// window pixel and plane.
    static func calibrationLine(_ pair: Pair, iterations: Int) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { throw SubtitleUpscaler.UpscaleError.unavailable("no device") }
        var line = "  \(pair.name)"
        for shape in shapes {
            let model = try model(pair, shape)
            let surfaces = try Surfaces(shape: shape, device: device)
            var samples: [Double] = []
            for i in 0..<(iterations + 3) {
                let t0 = DispatchTime.now().uptimeNanoseconds
                try predict(model, surfaces)
                if i >= 3 { samples.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6) }
            }
            let t = SubtitleUpscaler.Spread(samples).p50
            line += String(format: "  %@ %5.2f (%4.1f)", shape.function as NSString, t, t * 1e6 / Double(2 * shape.height * shape.width))
        }
        return line
    }
}

extension SubtitleUpscaleNeuralEngine {
    /// A cold load of every function of `pair`: the compiled model copied to a path the system has not seen, so its
    /// Neural Engine compile is not cached, then each function loaded and run once, one after another or each on a
    /// thread of its own. Wall time in all and each function's own.
    static func coldLoadLine(_ pair: Pair, parallel: Bool, strategy: MLOptimizationHints.SpecializationStrategy) throws -> String {
        let fm = FileManager.default
        let source = try ArtCNNCoreMLModels.url(alpha: pair.alpha, luma: pair.luma)
        let root = fm.temporaryDirectory.appendingPathComponent("LibMLSubUpscaler-cold-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let url = root.appendingPathComponent(source.lastPathComponent, isDirectory: true)
        try fm.copyItem(at: source, to: url)
        final class Loads: @unchecked Sendable {
            let lock = NSLock()
            var models: [MLModel?]
            var each: [Double]
            var failure: Error?
            init(_ n: Int) { models = Array(repeating: nil, count: n); each = Array(repeating: 0, count: n) }
        }
        let loads = Loads(shapes.count)
        @Sendable func run(_ k: Int) {
            let t0 = DispatchTime.now().uptimeNanoseconds
            do {
                let shape = shapes[k]
                let model = try load(shape, from: url, strategy: strategy)
                loads.lock.withLock { loads.models[k] = model }
                try runOnce(model, shape)
            } catch {
                loads.lock.withLock { loads.failure = error }
            }
            let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
            loads.lock.withLock { loads.each[k] = ms }
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        if parallel {
            let group = DispatchGroup()
            for k in shapes.indices {
                group.enter()
                Thread { run(k); group.leave() }.start()
            }
            group.wait()
        } else {
            for k in shapes.indices { run(k) }
        }
        let total = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        if let failure = loads.failure { throw failure }
        let models = loads.models, each = loads.each
        let name = strategy == .fastPrediction ? "fastPrediction" : "default"
        var line = String(format: "  %@ %@ %@: total %.0f |", pair.name as NSString, parallel ? "parallel" : "serial  ",
                          name as NSString, total)
        for (shape, ms) in zip(shapes, each) { line += String(format: " %@ %.0f", shape.function as NSString, ms) }
        // Then each loaded function's prediction on its own surfaces, as the upscaler runs it, median.
        guard let device = MTLCreateSystemDefaultDevice() else { return line }
        line += "\n      predict p50:"
        for (shape, model) in zip(shapes, models) {
            guard let model else { continue }
            let surfaces = try Surfaces(shape: shape, device: device)
            var samples: [Double] = []
            for i in 0..<23 {
                let t0 = DispatchTime.now().uptimeNanoseconds
                try predict(model, surfaces)
                if i >= 3 { samples.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6) }
            }
            line += String(format: " %@ %.2f", shape.function as NSString, SubtitleUpscaler.Spread(samples).p50)
        }
        return line
    }

    /// Cold loads of the balanced pairing, serial and parallel, with each specialisation strategy.
    public static func coldLoadReport() -> String {
        var lines = ["Cold Neural Engine loads (fresh model path), wall ms"]
        let pair = SubtitleUpscaler.Configuration.balanced.pair
        for strategy in [MLOptimizationHints.SpecializationStrategy.default, .fastPrediction] {
            for parallel in [false, true] {
                do { lines.append(try coldLoadLine(pair, parallel: parallel, strategy: strategy)) }
                catch { lines.append("  failed: \(error)") }
            }
        }
        return lines.joined(separator: "\n")
    }
}
