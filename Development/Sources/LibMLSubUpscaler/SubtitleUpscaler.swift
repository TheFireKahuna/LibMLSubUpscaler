import CoreGraphics
import CoreVideo
import Foundation
import Metal
import os

/// Upscales a bitmap subtitle (PGS, DVB, DVD, a display set's RGBA picture) 2x with ArtCNN on the Neural Engine,
/// leaving the GPU to the video (``SubtitleUpscaleNeuralEngine``; GPU passes prepare its input and composite its output).
///
/// ArtCNN is a single-channel model, so the picture is split into its alpha and its premultiplied luma, each plane is
/// run through the network, and the colour comes back as the source's chroma interpolated bilinearly (subtitle
/// colour is near-constant across a glyph: the shape lives in alpha and luma) and scaled by the new alpha. The result
/// is premultiplied, alpha clamped to [0, 1] and colour to [0, alpha], so the network's overshoot cannot paint a halo.
/// Output pixels farther than 5 source pixels from any non-transparent pixel are exactly transparent, so the
/// network runs only on the content bounds grown by that support (``SubtitleUpscaleTilePlan``).
///
/// One process-wide device, pipeline set and model set (``prewarm(_:)``); the GPU passes go through the caller's
/// ``SubtitleUpscaleSubmission``. Each upscaler keeps its working buffers and window surfaces between calls, so the
/// steady state allocates nothing. An instance runs one
/// picture at a time (calls are serialised); ``upscale(_:)`` runs a burst and waits.
public final class SubtitleUpscaler: @unchecked Sendable {

    /// How the source's colour relates to its alpha. The engine's bitmap subtitles are premultiplied.
    /// Frozen: a source's colour is premultiplied by its alpha or it is not.
    @frozen public enum SourceAlpha: Sendable, Hashable {
        case premultiplied
        case straight
    }

    public struct Configuration: Sendable, Hashable {
        /// The network on the alpha plane: the outline's outer edge, against the video.
        public var alphaModel: ArtCNNModel
        /// The network on the premultiplied luma plane: the fill's edge against the outline, the highest-contrast
        /// edge of a light subtitle with a dark outline.
        public var lumaModel: ArtCNNModel

        public init(alphaModel: ArtCNNModel = .c4f32, lumaModel: ArtCNNModel = .c4f32) {
            self.alphaModel = alphaModel
            self.lumaModel = lumaModel
        }

        /// C4F32 on both planes.
        public static let `default` = Configuration()
        public static let c4f32 = Configuration()
        /// C4F32 on luma, C4F16 on alpha, about two thirds of C4F32's work: the colour stays C4F32's (a mean
        /// difference of 0.01 8-bit codes on glyph strips), the outline's outer edge takes C4F16's shape (a mean
        /// 0.8 codes of alpha). The other split, C4F32 on alpha, costs the same and changes the colour as much as
        /// C4F16 everywhere does, because the fill edge is where a subtitle's contrast is.
        public static let balanced = Configuration(alphaModel: .c4f16, lumaModel: .c4f32)
        public static let c4f16 = Configuration(alphaModel: .c4f16, lumaModel: .c4f16)

        var pair: SubtitleUpscaleNeuralEngine.Pair { .init(alpha: alphaModel, luma: lumaModel) }
    }

    public enum UpscaleError: Error, CustomStringConvertible {
        case unavailable(String)
        case size(String)
        case gpu(String)
        public var description: String {
            switch self {
            case .unavailable(let s): "subtitle upscaler unavailable: \(s)"
            case .size(let s): "subtitle upscaler size: \(s)"
            case .gpu(let s): "subtitle upscaler GPU error: \(s)"
            }
        }
    }

    /// One picture to upscale: `source` (any float-readable colour texture: BGRA8, RGBA8, RGBA16F) into
    /// `destination` (writable, any size; see ``upscale(_:)``).
    public struct Job {
        public var source: MTLTexture
        public var destination: MTLTexture
        public var sourceAlpha: SourceAlpha
        /// Where the source's non-transparent pixels are (``Content``); every other source pixel is fully
        /// transparent. The network runs only near them; nil runs it on the whole picture. The output is the same
        /// either way as long as the promise holds.
        public var content: Content?
        public init(source: MTLTexture, destination: MTLTexture, sourceAlpha: SourceAlpha = .premultiplied, content: Content? = nil) {
            self.source = source
            self.destination = destination
            self.sourceAlpha = sourceAlpha
            self.content = content
        }
    }

    /// Where a picture's non-transparent pixels are, row by row: for each row of `bounds`, the columns from its first
    /// such pixel to its last, nil for a row with none. A caller that knows only rectangles (a display set's object
    /// rectangles) gives them as ``init(bounds:)``, every row in full; ``scan(_:width:height:bytesPerRow:alphaOffset:)``
    /// reads the rows of a bitmap. The upscaler plans its windows from the rows, so two lines of different lengths,
    /// and the gap between them, cost only what they hold.
    public struct Content: Sendable, Equatable {
        /// Integral, picture coordinates; empty when there is no content.
        public let bounds: CGRect
        /// One per row of `bounds`: the first and last columns (picture coordinates) of content in that row.
        let spans: [ClosedRange<Int>?]

        public static let empty = Content(bounds: .zero)

        public init(bounds: CGRect) {
            let b = bounds.integral
            guard !b.isNull, !b.isEmpty else { self.bounds = .zero; spans = []; return }
            self.bounds = b
            spans = Array(repeating: Int(b.minX)...Int(b.maxX) - 1, count: Int(b.height))
        }

        init(bounds: CGRect, spans: [ClosedRange<Int>?]) {
            self.bounds = bounds
            self.spans = spans
        }

        /// The rows of the pixels with non-zero alpha in a 4-byte-per-pixel bitmap, nil when there are none.
        /// `alphaOffset` is the alpha byte's position in the pixel (3 for RGBA and BGRA).
        public static func scan(_ base: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int, alphaOffset: Int) -> Content? {
            var rows = [ClosedRange<Int>?](repeating: nil, count: height)
            var x0 = width, x1 = -1, y0 = height, y1 = -1
            for y in 0..<height {
                let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
                var first = -1, last = -1
                for x in 0..<width where row[4 * x + alphaOffset] != 0 {
                    if first < 0 { first = x }
                    last = x
                }
                guard first >= 0 else { continue }
                rows[y] = first...last
                x0 = min(x0, first); x1 = max(x1, last)
                if y0 == height { y0 = y }
                y1 = y
            }
            guard x1 >= 0 else { return nil }
            return Content(bounds: CGRect(x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1), spans: Array(rows[y0...y1]))
        }

        /// The content columns of picture row `y`.
        func span(row y: Int) -> ClosedRange<Int>? {
            let i = y - Int(bounds.minY)
            return i >= 0 && i < spans.count ? spans[i] : nil
        }

        /// The kept area of a `width` x `height` picture: the pixels within `support` of the content bounds, which
        /// hold every output pixel that is not exactly transparent. Empty without content.
        func kept(width: Int, height: Int, support: Int) -> Rect {
            let picture = CGRect(x: 0, y: 0, width: width, height: height)
            let c = bounds.intersection(picture)
            guard !c.isNull, !c.isEmpty else { return Rect(x: 0, y: 0, width: 0, height: 0) }
            let k = c.insetBy(dx: -CGFloat(support), dy: -CGFloat(support)).intersection(picture)
            return Rect(x: Int(k.minX), y: Int(k.minY), width: Int(k.width), height: Int(k.height))
        }

        /// The content of a 2x pass's output: every output pixel within `support` of this content (the only ones
        /// not exactly transparent), row by row.
        func doubled(support: Int, width: Int, height: Int) -> Content {
            guard !spans.isEmpty else { return .empty }
            let y0 = max(0, Int(bounds.minY) - support), y1 = min(height, Int(bounds.maxY) + support)
            var rows: [ClosedRange<Int>?] = []
            for y in y0..<y1 {
                var lo = Int.max, hi = -1
                for yy in max(y0, y - support)...min(y1 - 1, y + support) {
                    if let s = span(row: yy) { lo = min(lo, s.lowerBound); hi = max(hi, s.upperBound) }
                }
                let out: ClosedRange<Int>? = hi < 0 ? nil : 2 * max(0, lo - support)...2 * min(width - 1, hi + support) + 1
                rows.append(out); rows.append(out)
            }
            let xs = rows.compactMap { $0 }
            guard !xs.isEmpty else { return .empty }
            let x0 = xs.map(\.lowerBound).min()!, x1 = xs.map(\.upperBound).max()!
            return Content(bounds: CGRect(x: x0, y: 2 * y0, width: x1 - x0 + 1, height: rows.count), spans: rows)
        }
    }

    public var configuration: Configuration {
        get { lock.withLock { _configuration } }
        set { lock.withLock { _configuration = newValue } }
    }

    /// The shared Metal device the upscaler runs on: textures passed in must come from it.
    public var device: MTLDevice { kernels.device }
    let kernels: SubtitleUpscaleKernels
    /// The caller's submission, which every GPU pass goes through.
    let submission: any SubtitleUpscaleSubmission
    /// Serialises every call; recursive, so a convenience holds it across its staging and its ``upscale(_:)``.
    private let lock = NSRecursiveLock()
    var _configuration: Configuration
    var workspace: Workspace

    /// Wall milliseconds of the last pass's stages, for the benchmark.
    struct NeuralEngineStages {
        var plan = 0.0
        var gather = 0.0
        var predict = 0.0
        var composite = 0.0
        var tiles = 0
    }
    var lastNeuralEngineStages = NeuralEngineStages()

    /// Output pixels farther than this many source pixels from content are transparent (1...7; see the shader's
    /// notes on the cut).
    var supportRadius = 5 {
        didSet { precondition((1...7).contains(supportRadius)) }
    }

    /// Has `submission` make the shared pipelines, loads `configuration`'s Core ML model and has the Neural Engine
    /// compile each function, on a background queue, so the first picture waits for none of it; then replaces each
    /// function with its `fastPrediction` specialisation behind them, a function at a time. For app open: a second or
    /// two after an install, an OS update or a purged cache (the replacement continues in the background for some
    /// seconds more), well under a second otherwise.
    public static func prewarm(_ configuration: Configuration = .default, submission: any SubtitleUpscaleSubmission) {
        DispatchQueue.global(qos: .utility).async {
            _ = try? SubtitleUpscaleKernels.get(submission)
            _ = try? SubtitleUpscaleNeuralEngine.prewarm(configuration.pair)
        }
    }

    /// An upscaler whose GPU passes go through `submission`, which must be on the device the shared pipelines were
    /// made on (``device``: the first submission's).
    public init(configuration: Configuration = .default, submission: any SubtitleUpscaleSubmission) throws {
        kernels = try SubtitleUpscaleKernels.get(submission)
        guard submission.device === kernels.device else {
            throw UpscaleError.unavailable("the submission is on \(submission.device.name), not the upscaler's device")
        }
        self.submission = submission
        _configuration = configuration
        workspace = Workspace(device: kernels.device)
    }

    /// Readies the upscaler for its first subtitle: the configuration's models loaded (what ``prewarm(_:)`` does,
    /// waited for), this upscaler's window surfaces made and each run through its model once, then one small
    /// picture through the whole pass, so the first subtitle pays only the steady-state cost. The surfaces are one
    /// set per window shape, and as many sets of each shape as the plans of representative subtitles on `canvas` use
    /// (one to three lines across 30 to 90% of its width, a line a twentieth of its height). Blocks: for a
    /// background thread, when a session starts. A plan needing more windows of one shape than this made still makes
    /// them on first use.
    public func prepare(canvas: CGSize = CGSize(width: 1920, height: 1080)) throws {
        try lock.withLock {
            let config = _configuration
            _ = try SubtitleUpscaleKernels.get(submission)
            try SubtitleUpscaleNeuralEngine.prewarm(config.pair)
            var counts = Dictionary(uniqueKeysWithValues: SubtitleUpscaleNeuralEngine.shapes.map { ($0, 1) })
            let lineHeight = max(8, Int(canvas.height / 20)), gap = lineHeight / 4
            for lines in 1...3 {
                for fraction in [0.3, 0.6, 0.9] {
                    let w = max(16, Int(canvas.width * fraction)), h = lines * lineHeight + (lines - 1) * gap
                    var spans = [ClosedRange<Int>?](repeating: nil, count: h)
                    for k in 0..<lines {
                        // Each line a little shorter than the one above, as subtitle lines run.
                        let lw = max(8, w - k * w / 5), x0 = (w - lw) / 2
                        for y in (k * (lineHeight + gap))..<(k * (lineHeight + gap) + lineHeight) { spans[y] = x0...(x0 + lw - 1) }
                    }
                    let content = Content(bounds: CGRect(x: 0, y: 0, width: w, height: h), spans: spans)
                    let plan = SubtitleUpscaleTilePlan.plan(content: content, width: w, height: h, support: supportRadius,
                                                            alpha: config.alphaModel, luma: config.lumaModel)
                    var used: [SubtitleUpscaleNeuralEngine.Shape: Int] = [:]
                    for window in plan.windows { used[.init(height: window.height, width: window.width), default: 0] += 1 }
                    for (shape, n) in used { counts[shape] = max(counts[shape] ?? 0, n) }
                }
            }
            let windows = counts.flatMap { shape, n in
                Array(repeating: SubtitleUpscaleTilePlan.Window(x: 0, y: 0, width: shape.width, height: shape.height), count: n)
            }
            for s in try workspace.surfaces.surfaces(for: windows) {
                try SubtitleUpscaleNeuralEngine.predict(try SubtitleUpscaleNeuralEngine.model(config.pair, s.shape), s)
            }
            // One opaque pixel: the GPU passes' first dispatches and the mask buffers.
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 16, height: 16, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .shared
            guard let source = device.makeTexture(descriptor: d) else { throw UpscaleError.unavailable("no texture") }
            var pixel: [UInt8] = [255, 255, 255, 255]
            source.replace(region: MTLRegionMake2D(8, 8, 1, 1), mipmapLevel: 0, withBytes: &pixel, bytesPerRow: 4)
            d.width = 32; d.height = 32
            d.storageMode = .private
            guard let destination = device.makeTexture(descriptor: d) else { throw UpscaleError.unavailable("no texture") }
            try upscale([Job(source: source, destination: destination, content: Content(bounds: CGRect(x: 8, y: 8, width: 1, height: 1)))])
        }
    }

    /// Upscales each job and waits. A destination twice its source takes one 2x pass; any other size takes 2x passes
    /// (one, or two when the destination is at least 4x) then a Lanczos-3 resample to the exact size, antialiased
    /// when it reduces.
    public func upscale(_ jobs: [Job]) throws {
        try lock.withLock {
            for job in jobs {
                let plan = try scalePlan(job)
                for pass in plan.passes { try passLocked(pass) }
                if let (src, dst) = plan.resample { try run { try encodeResample(src, dst, into: $0) } }
            }
        }
    }

    /// The 2x passes `job` takes, into the workspace's intermediates where it needs more than one, and the final
    /// resample (source, destination) when the destination is not exactly 2x.
    private func scalePlan(_ job: Job) throws -> (passes: [Job], resample: (MTLTexture, MTLTexture)?) {
        let sw = job.source.width, sh = job.source.height
        let dw = job.destination.width, dh = job.destination.height
        if dw == 2 * sw, dh == 2 * sh { return ([job], nil) }
        let scale = max(Double(dw) / Double(sw), Double(dh) / Double(sh))
        var passes: [Job] = []
        var current = job.source
        var alpha = job.sourceAlpha
        var content = job.content
        for pass in 0..<(scale >= 4 ? 2 : 1) {
            let target = try workspace.intermediate(pass, width: current.width * 2, height: current.height * 2)
            passes.append(Job(source: current, destination: target, sourceAlpha: alpha, content: content))
            content = (content ?? Content(bounds: CGRect(x: 0, y: 0, width: current.width, height: current.height)))
                .doubled(support: supportRadius, width: current.width, height: current.height)
            current = target
            alpha = .premultiplied
        }
        return (passes, (current, job.destination))
    }

    private func encodeResample(_ src: MTLTexture, _ dst: MTLTexture, into pass: any SubtitleUpscalePass) throws {
        let across = try workspace.intermediate(2, width: dst.width, height: src.height)
        pass.setPipeline(kernels.resample)
        for (axis, (from, to)) in [(src, across), (across, dst)].enumerated() {
            // The second axis reads what the first wrote.
            if axis > 0 { pass.barrier() }
            pass.setTexture(from, index: 0)
            pass.setTexture(to, index: 1)
            pass.setValue(UInt32(axis), index: 0)
            dispatch(pass, kernels.resample, width: to.width, height: to.height)
        }
    }

    // MARK: - Conveniences

    /// Upscales `pixelBuffer` (32BGRA) 2x, or to `size`, into a new 32BGRA buffer, and waits.
    public func upscale(_ pixelBuffer: CVPixelBuffer, sourceAlpha: SourceAlpha = .premultiplied, size: CGSize? = nil) throws -> CVPixelBuffer {
        try lock.withLock { try upscaleLocked(pixelBuffer, sourceAlpha: sourceAlpha, size: size) }
    }

    private func upscaleLocked(_ pixelBuffer: CVPixelBuffer, sourceAlpha: SourceAlpha, size: CGSize?) throws -> CVPixelBuffer {
        let w = CVPixelBufferGetWidth(pixelBuffer), h = CVPixelBufferGetHeight(pixelBuffer)
        let dw = size.map { Int($0.width.rounded()) } ?? 2 * w
        let dh = size.map { Int($0.height.rounded()) } ?? 2 * h
        var out: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any],
                                           kCVPixelBufferMetalCompatibilityKey: true]
        guard CVPixelBufferCreate(nil, dw, dh, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &out) == kCVReturnSuccess,
              let out else { throw UpscaleError.unavailable("no output pixel buffer") }
        let source = try workspace.texture(for: pixelBuffer), destination = try workspace.texture(for: out)
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        let content = CVPixelBufferGetBaseAddress(pixelBuffer).map {
            Content.scan($0, width: w, height: h, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer), alphaOffset: 3)
        } ?? nil
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        try upscale([Job(source: source.texture, destination: destination.texture, sourceAlpha: sourceAlpha, content: content ?? .empty)])
        withExtendedLifetime((source, destination)) {}
        return out
    }

    /// Upscales `image` 2x, or to `size`, and waits. The result is RGBA, premultiplied last, sRGB.
    public func upscale(_ image: CGImage, size: CGSize? = nil) throws -> CGImage {
        try lock.withLock { try upscaleLocked(image, size: size) }
    }

    private func upscaleLocked(_ image: CGImage, size: CGSize?) throws -> CGImage {
        let w = image.width, h = image.height
        let dw = size.map { Int($0.width.rounded()) } ?? 2 * w
        let dh = size.map { Int($0.height.rounded()) } ?? 2 * h
        let source = try workspace.staging(0, width: w, height: h)
        let destination = try workspace.staging(1, width: dw, height: dh)
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        var content: Content?
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let info = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        try pixels.withUnsafeMutableBytes { raw in
            guard let context = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                          space: space, bitmapInfo: info) else { throw UpscaleError.unavailable("no bitmap context") }
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            source.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: w * 4)
            content = Content.scan(raw.baseAddress!, width: w, height: h, bytesPerRow: w * 4, alphaOffset: 3)
        }
        try upscale([Job(source: source, destination: destination, content: content ?? .empty)])
        var result = [UInt8](repeating: 0, count: dw * dh * 4)
        result.withUnsafeMutableBytes { raw in
            destination.getBytes(raw.baseAddress!, bytesPerRow: dw * 4, from: MTLRegionMake2D(0, 0, dw, dh), mipmapLevel: 0)
        }
        guard let provider = CGDataProvider(data: Data(result) as CFData),
              let made = CGImage(width: dw, height: dh, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: dw * 4, space: space,
                                 bitmapInfo: CGBitmapInfo(rawValue: info), provider: provider, decode: nil, shouldInterpolate: true,
                                 intent: .defaultIntent) else { throw UpscaleError.unavailable("no output image") }
        return made
    }

    /// Runs `encode` in a batch of its own and waits for it. A batch whose encoding throws is still committed (with
    /// what it has), so it is never left open.
    func run(_ encode: (any SubtitleUpscalePass) throws -> Void) throws {
        guard let batch = submission.makeBatch() else { throw UpscaleError.gpu("no batch") }
        let waiter = Completions()
        guard let pass = batch.makePass() else {
            waiter.commit(batch)
            _ = waiter.wait()
            throw UpscaleError.gpu("no compute pass")
        }
        var thrown: Error?
        do { try encode(pass) } catch { thrown = error }
        waiter.commit(batch)
        let failure = waiter.wait()
        if let thrown { throw thrown }
        if let failure { throw UpscaleError.gpu(failure) }
    }

    /// The completions of a pass's batches: each commit counted, the first failure kept, and one wait for them all.
    final class Completions: @unchecked Sendable {
        private let group = DispatchGroup()
        private let failure = OSAllocatedUnfairLock<String?>(initialState: nil)

        func commit(_ batch: any SubtitleUpscaleBatch) {
            group.enter()
            batch.commit { [self] why in
                if let why { failure.withLock { if $0 == nil { $0 = why } } }
                group.leave()
            }
        }

        /// Waits for every batch committed so far; the first failure, or nil.
        func wait() -> String? {
            group.wait()
            return failure.withLock { $0 }
        }
    }

    // MARK: - One picture

    /// Mirrors `UpGeom` in the shader.
    struct Geometry {
        var width: UInt32 = 0
        var height: UInt32 = 0
        var wordsX: UInt32 = 0
        var straightAlpha: UInt32 = 0
        var originX: Int32 = 0
        var originY: Int32 = 0
        var pictureWidth: UInt32 = 0
        var pictureHeight: UInt32 = 0
        var support: UInt32 = 0
    }

    /// Encodes the kept area's content mask (the support the composite keeps). Nil when the kept area is empty.
    private func encodeMask(_ job: Job, kept: Rect, pass: any SubtitleUpscalePass) throws -> Geometry? {
        let pw = job.source.width, ph = job.source.height
        var g = Geometry()
        g.support = UInt32(supportRadius)
        g.originX = Int32(kept.x); g.originY = Int32(kept.y)
        g.pictureWidth = UInt32(pw); g.pictureHeight = UInt32(ph)
        g.width = UInt32(kept.width); g.height = UInt32(kept.height)
        g.straightAlpha = job.sourceAlpha == .straight ? 1 : 0
        guard kept.width > 0 else { return nil }
        let h = kept.height
        let wordsX = (kept.width + 31) / 32
        g.wordsX = UInt32(wordsX)
        try workspace.reserveMask(height: h, wordsX: wordsX)
        let ws = workspace
        // Bits, then their row dilation, then the square one: each step reads the one before.
        pass.setTexture(job.source, index: 0)
        pass.setPipeline(kernels.maskBits)
        pass.setBuffer(ws.bits, offset: 0, index: 0)
        pass.setValue(g, index: 1)
        // 32 wide, so each simdgroup is one word of one row.
        pass.dispatchThreads(MTLSize(width: wordsX * 32, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        pass.barrier()
        pass.setPipeline(kernels.maskRows)
        pass.setBuffer(ws.bits, offset: 0, index: 0)
        pass.setBuffer(ws.rows, offset: 0, index: 1)
        pass.setValue(g, index: 2)
        dispatch(pass, kernels.maskRows, width: wordsX, height: h)
        pass.barrier()
        pass.setPipeline(kernels.maskColumns)
        pass.setBuffer(ws.rows, offset: 0, index: 0)
        pass.setBuffer(ws.support, offset: 0, index: 1)
        pass.setValue(g, index: 2)
        dispatch(pass, kernels.maskColumns, width: wordsX, height: h)
        return g
    }

    // MARK: - One pass

    /// A rectangle in picture coordinates.
    struct Rect: Equatable {
        var x: Int
        var y: Int
        var width: Int
        var height: Int
    }

    /// Mirrors `UpWindow` in the shader.
    struct WindowRecord {
        var x: Int32
        var y: Int32
        var coreX: Int32
        var coreY: Int32
        var coreWidth: UInt32
        var coreHeight: UInt32
    }

    /// The source pixels no window's core covers, as rectangles (picture coordinates): rows outside every band,
    /// and each band's columns beside its cores. All of them are exactly transparent in the output.
    static func uncovered(_ plan: SubtitleUpscaleTilePlan, kept: Rect, width: Int, height: Int) -> [Rect] {
        var rects: [Rect] = []
        var y = 0
        for band in plan.bands {
            let y0 = max(band.y, kept.y), y1 = min(band.y + band.height, kept.y + kept.height)
            let x0 = max(band.x, kept.x), x1 = min(band.x + band.width, kept.x + kept.width)
            guard y1 > y0 else { continue }
            if y0 > y { rects.append(Rect(x: 0, y: y, width: width, height: y0 - y)) }
            if x0 > 0 { rects.append(Rect(x: 0, y: y0, width: x0, height: y1 - y0)) }
            if x1 < width { rects.append(Rect(x: x1, y: y0, width: width - x1, height: y1 - y0)) }
            y = y1
        }
        if y < height { rects.append(Rect(x: 0, y: y, width: width, height: height - y)) }
        return rects
    }

    /// One 2x pass: the mask and the window gather on the GPU, a prediction per window and plane on the Neural
    /// Engine, then the composite on the GPU, each window's planes shared between them through its IOSurfaces. Waits
    /// for each step.
    private func passLocked(_ job: Job) throws {
        let config = _configuration
        let pw = job.source.width, ph = job.source.height
        guard pw > 0, ph > 0 else { throw UpscaleError.size("empty source") }
        guard job.destination.width == 2 * pw, job.destination.height == 2 * ph else {
            throw UpscaleError.size("destination \(job.destination.width)x\(job.destination.height) is not 2x \(pw)x\(ph)")
        }
        var stages = NeuralEngineStages()
        var mark = DispatchTime.now().uptimeNanoseconds
        func lap() -> Double {
            let now = DispatchTime.now().uptimeNanoseconds
            defer { mark = now }
            return Double(now - mark) / 1e6
        }
        defer { lastNeuralEngineStages = stages }
        let content = job.content ?? Content(bounds: CGRect(x: 0, y: 0, width: pw, height: ph))
        let kept = content.kept(width: pw, height: ph, support: supportRadius)
        let plan = SubtitleUpscaleTilePlan.plan(content: content, width: pw, height: ph, support: supportRadius,
                                                alpha: config.alphaModel, luma: config.lumaModel)
        let clears = Self.uncovered(plan, kept: kept, width: pw, height: ph)
        let surfaces = try workspace.surfaces.surfaces(for: plan.windows)
        let records = plan.windows.map { w in
            let core = CGRect(x: w.x + SubtitleUpscaleTilePlan.reach, y: w.y + SubtitleUpscaleTilePlan.reach,
                              width: w.width - 2 * SubtitleUpscaleTilePlan.reach, height: w.height - 2 * SubtitleUpscaleTilePlan.reach)
                .intersection(CGRect(x: kept.x, y: kept.y, width: kept.width, height: kept.height))
            return WindowRecord(x: Int32(w.x), y: Int32(w.y), coreX: Int32(core.minX), coreY: Int32(core.minY),
                                coreWidth: UInt32(core.width), coreHeight: UInt32(core.height))
        }
        var g = Geometry()
        stages.plan = lap()
        // The clears and the mask, then a batch per window's gather that signals its window's event, then a batch
        // per window's composite that waits on its window's other event, which the CPU signals when that window's
        // prediction returns: every batch goes to the GPU before any prediction runs, and each composite starts as
        // its window's outputs are ready while later windows are still predicting. Batches from one submission run
        // in the order they are committed.
        let completions = Completions()
        guard let prepare = submission.makeBatch(), let pass = prepare.makePass() else { throw UpscaleError.gpu("no batch") }
        pass.setPipeline(kernels.clearRect)
        pass.setTexture(job.destination, index: 0)
        for rect in clears {
            pass.setValue(SIMD4<Int32>(Int32(rect.x), Int32(rect.y), Int32(rect.width), Int32(rect.height)), index: 0)
            dispatch(pass, kernels.clearRect, width: rect.width, height: rect.height)
        }
        let made = try encodeMask(job, kept: kept, pass: pass)
        completions.commit(prepare)
        guard let made, !records.isEmpty else {
            if let failure = completions.wait() { throw UpscaleError.gpu(failure) }
            return
        }
        g = made
        let events = try workspace.events(records.count)
        let value = workspace.nextEventValue()
        // Whatever happens below, every compose event is signalled, so no queued composite waits forever.
        var signalled = false
        func release() {
            guard !signalled else { return }
            signalled = true
            for e in events.compose.prefix(records.count) where e.signaledValue < value { e.signaledValue = value }
        }
        defer {
            release()
            _ = completions.wait()
        }
        // Each window's gather in a batch of its own, then its gather event: its prediction starts as its inputs are
        // written, while later windows are still gathering.
        for (k, record) in records.enumerated() {
            guard let batch = submission.makeBatch(), let pass = batch.makePass() else { throw UpscaleError.gpu("no batch") }
            pass.setPipeline(kernels.tileGather)
            pass.setTexture(job.source, index: 0)
            pass.setValue(g, index: 0)
            pass.setTexture(surfaces[k].alphaIn.texture, index: 1)
            pass.setTexture(surfaces[k].lumaIn.texture, index: 2)
            pass.setValue(record, index: 1)
            pass.dispatchThreads(MTLSize(width: plan.windows[k].width, height: plan.windows[k].height, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
            batch.signal(events.gather[k], value: value)
            completions.commit(batch)
        }
        for (k, record) in records.enumerated() where record.coreWidth > 0 && record.coreHeight > 0 {
            guard let batch = submission.makeBatch() else { throw UpscaleError.gpu("no batch") }
            batch.wait(for: events.compose[k], value: value)
            guard let pass = batch.makePass() else {
                completions.commit(batch)
                throw UpscaleError.gpu("no compute pass")
            }
            pass.setPipeline(kernels.compositeTiles)
            pass.setTexture(job.source, index: 0)
            pass.setTexture(job.destination, index: 1)
            pass.setBuffer(workspace.support, offset: 0, index: 0)
            pass.setValue(g, index: 1)
            pass.setTexture(surfaces[k].alphaOut.texture, index: 2)
            pass.setTexture(surfaces[k].lumaOut.texture, index: 3)
            pass.setValue(record, index: 2)
            dispatch(pass, kernels.compositeTiles, width: Int(record.coreWidth), height: Int(record.coreHeight))
            completions.commit(batch)
        }
        stages.tiles = records.count

        // All windows' predictions in flight together, each starting when its window's gather event fires and
        // signalling its composite's event as it returns.
        let failure = OSAllocatedUnfairLock<Error?>(initialState: nil)
        let firstInputs = OSAllocatedUnfairLock<UInt64>(initialState: .max)
        let pair = config.pair
        nonisolated(unsafe) let windows = surfaces
        DispatchQueue.concurrentPerform(iterations: windows.count) { k in
            do {
                guard events.gather[k].wait(untilSignaledValue: value, timeoutMS: 2000) else {
                    throw UpscaleError.gpu("window \(k)'s gather did not finish")
                }
                let now = DispatchTime.now().uptimeNanoseconds
                firstInputs.withLock { $0 = min($0, now) }
                try SubtitleUpscaleNeuralEngine.predict(try SubtitleUpscaleNeuralEngine.model(pair, windows[k].shape), windows[k])
            } catch {
                failure.withLock { if $0 == nil { $0 = error } }
            }
            events.compose[k].signaledValue = value
        }
        signalled = true
        let inputs = firstInputs.withLock { $0 }
        if inputs != .max, inputs > mark {
            stages.gather = Double(inputs - mark) / 1e6
            mark = inputs
        }
        stages.predict = lap()
        let gpuFailure = completions.wait()
        stages.composite = lap()
        if let gpuFailure { throw UpscaleError.gpu(gpuFailure) }
        if let error = failure.withLock({ $0 }) { throw error }
    }

    private func dispatch(_ pass: any SubtitleUpscalePass, _ pipeline: MTLComputePipelineState, width: Int, height: Int) {
        guard width > 0, height > 0 else { return }
        let tw = pipeline.threadExecutionWidth
        let th = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / tw, 8))
        pass.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    }

    // MARK: - Working set

    /// The buffers one picture's passes use, grown to the largest picture seen and kept.
    final class Workspace {
        let device: MTLDevice
        /// The content mask, one bit per pixel of the kept area: `bits`, its row dilation `rows` and its square
        /// dilation `support`, by S.
        private(set) var bits: MTLBuffer!
        private(set) var rows: MTLBuffer!
        private(set) var support: MTLBuffer!
        /// The window surfaces, shared by Metal and Core ML.
        let surfaces: SubtitleUpscaleNeuralEngine.Surfaces.Pool
        /// Two events per window: the GPU signals the first when the window's gather is written and its prediction
        /// waits on it; the CPU signals the second when the prediction returns and its composite waits on it. Every
        /// pass signals a new value.
        private var gatherEvents: [MTLSharedEvent] = []
        private var composeEvents: [MTLSharedEvent] = []
        private var eventValue: UInt64 = 0
        private var maskCapacity = 0
        private var textures: [Int: MTLTexture] = [:]
        private var textureCache: CVMetalTextureCache?

        init(device: MTLDevice) {
            self.device = device
            surfaces = SubtitleUpscaleNeuralEngine.Surfaces.Pool(device: device)
        }

        private func make(_ length: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(length, 16), options: .storageModePrivate) else {
                throw UpscaleError.unavailable("no \(length)-byte buffer")
            }
            return b
        }

        func events(_ count: Int) throws -> (gather: [MTLSharedEvent], compose: [MTLSharedEvent]) {
            while gatherEvents.count < count {
                guard let g = device.makeSharedEvent(), let c = device.makeSharedEvent() else { throw UpscaleError.unavailable("no shared event") }
                gatherEvents.append(g)
                composeEvents.append(c)
            }
            return (gatherEvents, composeEvents)
        }

        func nextEventValue() -> UInt64 {
            eventValue += 1
            return eventValue
        }

        func reserveMask(height: Int, wordsX: Int) throws {
            let mask = height * wordsX
            if mask > maskCapacity {
                bits = try make(mask * 4)
                rows = try make(mask * 4)
                support = try make(mask * 4)
                maskCapacity = mask
            }
        }

        /// Scratch RGBA16F textures for ``upscale(_:)`` at other sizes: slots 0 and 1 the 2x passes, 2 the
        /// resample's first axis.
        func intermediate(_ slot: Int, width: Int, height: Int) throws -> MTLTexture {
            try texture(slot, width: width, height: height, format: .rgba16Float, storage: .private)
        }

        /// Shared RGBA8 textures for the CGImage path: slot 0 the source, 1 the destination.
        func staging(_ slot: Int, width: Int, height: Int) throws -> MTLTexture {
            try texture(10 + slot, width: width, height: height, format: .rgba8Unorm, storage: .shared)
        }

        private func texture(_ slot: Int, width: Int, height: Int, format: MTLPixelFormat, storage: MTLStorageMode) throws -> MTLTexture {
            if let t = textures[slot], t.width == width, t.height == height { return t }
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = storage
            guard let t = device.makeTexture(descriptor: d) else { throw UpscaleError.unavailable("no \(width)x\(height) texture") }
            textures[slot] = t
            return t
        }

        func texture(for pixelBuffer: CVPixelBuffer) throws -> (texture: MTLTexture, backing: CVMetalTexture) {
            if textureCache == nil {
                var cache: CVMetalTextureCache?
                CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
                textureCache = cache
            }
            guard let cache = textureCache, CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
                throw UpscaleError.unavailable("pixel buffer is not 32BGRA")
            }
            var texture: CVMetalTexture?
            let attributes = [kCVMetalTextureUsage: NSNumber(value: MTLTextureUsage([.shaderRead, .shaderWrite]).rawValue)] as CFDictionary
            guard CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, pixelBuffer, attributes, .bgra8Unorm,
                                                            CVPixelBufferGetWidth(pixelBuffer), CVPixelBufferGetHeight(pixelBuffer),
                                                            0, &texture) == kCVReturnSuccess,
                  let texture, let metal = CVMetalTextureGetTexture(texture) else {
                throw UpscaleError.unavailable("no texture for the pixel buffer")
            }
            return (metal, texture)
        }
    }
}
