import Testing
import Foundation
import CoreGraphics
import CoreVideo
import Metal
@testable import LibMLSubUpscaler

/// ``SubtitleUpscaler`` against a Float32 CPU reference of the same pipeline (``ArtCNNNetwork/reference``, then the
/// composite written out below), and its invariants: exact transparency away from content, the clamps, content bounds
/// that change nothing, every window shape and grid the planner makes.
@Suite(.serialized)
struct SubtitleUpscaleTests {

    // MARK: - Reference

    struct Image: Equatable {
        let width: Int
        let height: Int
        var pixels: [SIMD4<Float>]
    }

    static let luma = SIMD3<Float>(0.2126, 0.7152, 0.0722)

    static func premultiplied(_ p: SubtitleUpscaleFixtures.Picture, straight: Bool) -> [SIMD4<Float>] {
        (0..<(p.width * p.height)).map { i in
            let c = SIMD4<Float>(Float(p.rgba[4 * i]), Float(p.rgba[4 * i + 1]), Float(p.rgba[4 * i + 2]), Float(p.rgba[4 * i + 3])) / 255
            return straight ? SIMD4(c.x * c.w, c.y * c.w, c.z * c.w, c.w) : c
        }
    }

    /// The network's output for `plane` on a transparent canvas: run with 8 transparent pixels around it (more
    /// than its 7-pixel reach), so the picture's edges are not ArtCNN's zero padding, then cropped.
    static func canvasNetwork(_ model: ArtCNNModel, _ plane: [Float], _ w: Int, _ h: Int) -> [Float] {
        let m = 8, pw = w + 2 * m, ph = h + 2 * m
        var padded = [Float](repeating: 0, count: pw * ph)
        for y in 0..<h { for x in 0..<w { padded[(y + m) * pw + x + m] = plane[y * w + x] } }
        let out = ArtCNNNetwork.network(model).reference(plane: padded, width: pw, height: ph)
        var cropped = [Float](repeating: 0, count: 4 * w * h)
        for y in 0..<(2 * h) { for x in 0..<(2 * w) { cropped[y * 2 * w + x] = out[(y + 2 * m) * 2 * pw + x + 2 * m] } }
        return cropped
    }

    /// The upscaler's output computed on the CPU in Float32.
    static func reference(_ p: SubtitleUpscaleFixtures.Picture, configuration: SubtitleUpscaler.Configuration, straight: Bool = false,
                          support: Int = 5) -> Image {
        let w = p.width, h = p.height
        let src = premultiplied(p, straight: straight)
        let alphaPlane = src.map(\.w)
        let lumaPlane = src.map { ($0.x * luma.x + $0.y * luma.y) + $0.z * luma.z }
        let a = canvasNetwork(configuration.alphaModel, alphaPlane, w, h)
        let l = canvasNetwork(configuration.lumaModel, lumaPlane, w, h)
        var out = [SIMD4<Float>](repeating: .zero, count: 4 * w * h)
        func at(_ x: Int, _ y: Int) -> SIMD4<Float> { x >= 0 && x < w && y >= 0 && y < h ? src[y * w + x] : .zero }
        for y in 0..<h {
            for x in 0..<w {
                var supported = false
                search: for sy in max(y - support, 0)...min(y + support, h - 1) {
                    for sx in max(x - support, 0)...min(x + support, w - 1) where src[sy * w + sx].w > 0 { supported = true; break search }
                }
                guard supported else { continue }
                for j in 0..<2 {
                    for i in 0..<2 {
                        let dx = 2 * i - 1, dy = 2 * j - 1
                        let c = (9 * at(x, y) + 3 * at(x + dx, y) + 3 * at(x, y + dy) + at(x + dx, y + dy)) / 16
                        let rgb = SIMD3(c.x, c.y, c.z)
                        let chroma = c.w > 1.0 / 1024 ? (rgb - (rgb * luma).sum()) / c.w : SIMD3<Float>(repeating: 0)
                        let o = (2 * y + j) * 2 * w + 2 * x + i
                        let alpha = min(max(a[o], 0), 1)
                        let colour = SIMD3(repeating: min(max(l[o], 0), 1)) + alpha * chroma
                        let clamped = SIMD3(min(max(colour.x, 0), alpha), min(max(colour.y, 0), alpha), min(max(colour.z, 0), alpha))
                        out[o] = SIMD4(clamped, alpha)
                    }
                }
            }
        }
        return Image(width: 2 * w, height: 2 * h, pixels: out)
    }

    // MARK: - Running

    static func floatTexture(_ device: MTLDevice, _ w: Int, _ h: Int) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .shared
        return device.makeTexture(descriptor: d)!
    }

    static func read(_ t: MTLTexture) -> Image {
        var pixels = [SIMD4<Float>](repeating: .zero, count: t.width * t.height)
        pixels.withUnsafeMutableBytes {
            t.getBytes($0.baseAddress!, bytesPerRow: t.width * 16, from: MTLRegionMake2D(0, 0, t.width, t.height), mipmapLevel: 0)
        }
        return Image(width: t.width, height: t.height, pixels: pixels)
    }

    static func content(_ p: SubtitleUpscaleFixtures.Picture) -> SubtitleUpscaler.Content {
        p.rgba.withUnsafeBytes {
            SubtitleUpscaler.Content.scan($0.baseAddress!, width: p.width, height: p.height, bytesPerRow: p.width * 4, alphaOffset: 3)
        } ?? .empty
    }

    /// Upscales 2x through ``SubtitleUpscaler/upscale(_:)`` into a Float32 texture.
    static func run(_ picture: SubtitleUpscaleFixtures.Picture, _ upscaler: SubtitleUpscaler, content: SubtitleUpscaler.Content? = nil,
                    straight: Bool = false) throws -> Image {
        let job = SubtitleUpscaler.Job(source: SubtitleUpscaleFixtures.texture(picture, device: upscaler.device)!,
                                       destination: floatTexture(upscaler.device, 2 * picture.width, 2 * picture.height),
                                       sourceAlpha: straight ? .straight : .premultiplied, content: content)
        try upscaler.upscale([job])
        return read(job.destination)
    }

    struct Error: CustomStringConvertible {
        var maxAlpha: Float = 0, meanAlpha: Float = 0, maxColour: Float = 0, meanColour: Float = 0
        var description: String {
            String(format: "alpha max %.5f mean %.6f, colour max %.5f mean %.6f", maxAlpha, meanAlpha, maxColour, meanColour)
        }
    }

    static func error(_ a: Image, _ b: Image) -> Error {
        precondition(a.width == b.width && a.height == b.height)
        var e = Error()
        var sumA: Double = 0, sumC: Double = 0
        for (p, q) in zip(a.pixels, b.pixels) {
            let d = SIMD4<Float>(Swift.abs(p.x - q.x), Swift.abs(p.y - q.y), Swift.abs(p.z - q.z), Swift.abs(p.w - q.w))
            e.maxAlpha = max(e.maxAlpha, d.w)
            e.maxColour = max(e.maxColour, max(d.x, d.y, d.z))
            sumA += Double(d.w)
            sumC += Double(d.x + d.y + d.z) / 3
        }
        e.meanAlpha = Float(sumA / Double(a.pixels.count))
        e.meanColour = Float(sumC / Double(a.pixels.count))
        return e
    }

    static let strip = SubtitleUpscaleFixtures.glyphs(width: 300, height: 64, lines: ["Quiet, wasn't it?"], fontSize: 40)
    static let coloured = SubtitleUpscaleFixtures.glyphs(width: 160, height: 48, lines: ["Yellow 42"], fontSize: 34,
                                                         fill: (1, 0.9, 0.1), outline: 2)

    // MARK: - Tests

    @Test("the upscale matches the Float32 reference", arguments: [
        ("C4F16", SubtitleUpscaler.Configuration.c4f16),
        ("C4F32", SubtitleUpscaler.Configuration.c4f32),
        ("balanced", SubtitleUpscaler.Configuration.balanced),
    ])
    func matchesReference(_ name: String, _ configuration: SubtitleUpscaler.Configuration) throws {
        let upscaler = try SubtitleUpscaler(configuration: configuration, submission: Metal3Submission.shared)
        // A strip smaller than one window, one wider than a core, two lines of different lengths (bands of their own
        // widths, the gap between them cleared), and a canvas run on its content.
        let wide = SubtitleUpscaleFixtures.glyphs(width: 712, height: 132, lines: ["Wider than one core", "and taller"], fontSize: 44)
        let uneven = SubtitleUpscaleFixtures.glyphs(width: 900, height: 150, lines: ["The first line is the long one,", "short"], fontSize: 52)
        var canvas = SubtitleUpscaleFixtures.glyphs(width: 320, height: 200, lines: ["Bottom line"], fontSize: 30, baselineFromBottom: 20)
        for y in 10..<30 { for x in 290..<320 { canvas.rgba[(y * 320 + x) * 4 ..< (y * 320 + x) * 4 + 4] = [200, 200, 200, 255] } }
        for picture in [Self.strip, Self.coloured, wide, uneven, canvas] {
            let e = Self.error(try Self.run(picture, upscaler, content: Self.content(picture)), Self.reference(picture, configuration: configuration))
            print("[upscale] \(name) \(picture.width)x\(picture.height) vs fp32: \(e)")
            #expect(e.maxAlpha < 0.005 && e.maxColour < 0.005, "\(picture.width)x\(picture.height): \(e)")
            #expect(e.meanAlpha < 0.0005 && e.meanColour < 0.0005, "\(picture.width)x\(picture.height): \(e)")
        }
    }

    @Test("an empty picture upscales to exact transparency; a solid one matches the reference")
    func emptyAndSolid() throws {
        let upscaler = try SubtitleUpscaler(configuration: .c4f32, submission: Metal3Submission.shared)
        let empty = SubtitleUpscaleFixtures.Picture(width: 50, height: 20, rgba: [UInt8](repeating: 0, count: 50 * 20 * 4))
        #expect(try Self.run(empty, upscaler).pixels.allSatisfy { $0 == .zero })
        #expect(try Self.run(empty, upscaler, content: Self.content(empty)).pixels.allSatisfy { $0 == .zero })
        var solid = empty
        for i in 0..<(50 * 20) { solid.rgba[4 * i ..< 4 * i + 4] = [200, 120, 40, 255] }
        let e = Self.error(try Self.run(solid, upscaler), Self.reference(solid, configuration: .c4f32))
        print("[upscale] solid: \(e)")
        #expect(e.maxAlpha < 0.005 && e.maxColour < 0.005)
    }

    @Test("alpha stays in [0, 1] and premultiplied colour within alpha")
    func clamps() throws {
        let upscaler = try SubtitleUpscaler(submission: Metal3Submission.shared)
        // Hard-edged, high-contrast content is where the networks overshoot most.
        var picture = SubtitleUpscaleFixtures.Picture(width: 64, height: 32, rgba: [UInt8](repeating: 0, count: 64 * 32 * 4))
        for y in 8..<24 { for x in 10..<54 where (x / 3 + y / 3) % 2 == 0 { picture.rgba[(y * 64 + x) * 4 ..< (y * 64 + x) * 4 + 4] = [255, 255, 255, 255] } }
        for image in [try Self.run(picture, upscaler), try Self.run(Self.strip, upscaler)] {
            for p in image.pixels {
                #expect(p.w >= 0 && p.w <= 1)
                #expect(p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x <= p.w && p.y <= p.w && p.z <= p.w)
            }
        }
    }

    @Test("output away from content is exactly transparent")
    func transparentAwayFromContent() throws {
        // A full canvas with one strip of text, a lot of transparency, and content touching the right edge.
        var picture = SubtitleUpscaleFixtures.glyphs(width: 480, height: 270, lines: ["Over here", "and there"], fontSize: 30,
                                                     baselineFromBottom: 20)
        for y in 40..<60 { for x in 470..<480 { picture.rgba[(y * 480 + x) * 4 ..< (y * 480 + x) * 4 + 4] = [128, 128, 128, 128] } }
        let image = try Self.run(picture, try SubtitleUpscaler(configuration: .balanced, submission: Metal3Submission.shared))
        // Rows 70...160 of the source are more than 7 pixels from any content.
        for y in 140..<320 { for x in 0..<960 { #expect(image.pixels[y * 960 + x] == .zero) } }
    }

    @Test("the content confines the network without changing the output")
    func contentConfines() throws {
        var picture = SubtitleUpscaleFixtures.glyphs(width: 480, height: 270, lines: ["Lower third"], fontSize: 30, baselineFromBottom: 30)
        for y in 0..<12 { for x in 0..<20 { picture.rgba[(y * 480 + x) * 4 ..< (y * 480 + x) * 4 + 4] = [90, 0, 0, 180] } }
        let content = Self.content(picture)
        #expect(content.bounds.minX == 0 && content.bounds.minY == 0 && content.bounds.maxY > 230)
        let upscaler = try SubtitleUpscaler(submission: Metal3Submission.shared)
        let whole = try Self.run(picture, upscaler)
        // Row by row, as a rectangle, or the whole picture: one output.
        #expect(try Self.run(picture, upscaler, content: content) == whole)
        #expect(try Self.run(picture, upscaler, content: SubtitleUpscaler.Content(bounds: content.bounds)) == whole)
        // The text alone, the corner block left out: the block's output vanishes, the text's stays.
        let text = SubtitleUpscaler.Content(bounds: CGRect(x: 0, y: 200, width: 480, height: 70))
        let partial = try Self.run(picture, upscaler, content: text)
        for y in 0..<540 { for x in 0..<960 where y >= 2 * 200 - 20 { #expect(partial.pixels[y * 960 + x] == whole.pixels[y * 960 + x]) } }
        #expect(partial.pixels[0] == .zero && whole.pixels[0] != .zero)
        // No content: all transparent.
        #expect(try Self.run(picture, upscaler, content: .empty).pixels.allSatisfy { $0 == .zero })
    }

    @Test("small and odd sizes match the reference", arguments: [(1, 1), (9, 5), (37, 23), (257, 31), (8, 8), (33, 2)])
    func oddSizes(_ w: Int, _ h: Int) throws {
        let upscaler = try SubtitleUpscaler(configuration: .c4f16, submission: Metal3Submission.shared)
        var picture = SubtitleUpscaleFixtures.glyphs(width: w, height: h, lines: ["Ag"], fontSize: CGFloat(max(h, 4)))
        // Opaque corners, so every edge of the picture is exercised.
        for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] { picture.rgba[(y * w + x) * 4 ..< (y * w + x) * 4 + 4] = [255, 255, 255, 255] }
        let e = Self.error(try Self.run(picture, upscaler), Self.reference(picture, configuration: .c4f16))
        #expect(e.maxAlpha < 0.005 && e.maxColour < 0.005, "\(w)x\(h): \(e)")
    }

    @Test("a picture covered with mixed window shapes matches the Float32 reference")
    func mixedGrid() throws {
        let upscaler = try SubtitleUpscaler(configuration: .balanced, submission: Metal3Submission.shared)
        // Text, and blocks in the corners, so the kept area is the whole picture: wider than the widest core and taller
        // than the tallest, its cover mixes core sizes on both axes and reaches every edge.
        let w = 2120, h = 150
        var picture = SubtitleUpscaleFixtures.glyphs(width: w, height: h, lines: ["Mixed cores across a wide picture", "and two lines of text"],
                                                     fontSize: 56)
        for (x0, y0) in [(0, 0), (w - 12, 0), (0, h - 9), (w - 12, h - 9)] {
            for y in y0..<(y0 + 9) { for x in x0..<(x0 + 12) { picture.rgba[(y * w + x) * 4 ..< (y * w + x) * 4 + 4] = [255, 255, 255, 255] } }
        }
        let content = Self.content(picture)
        let plan = SubtitleUpscaleTilePlan.plan(content: content, width: w, height: h, support: 5, alpha: .c4f16, luma: .c4f32)
        #expect(Set(plan.windows.map(\.height)).count > 1 && Set(plan.windows.map(\.width)).count > 1, "plan \(plan.windows)")
        let e = Self.error(try Self.run(picture, upscaler, content: content), Self.reference(picture, configuration: .balanced))
        print("[upscale] balanced \(w)x\(h) (\(plan.windows.count) windows) vs fp32: \(e)")
        #expect(e.maxAlpha < 0.005 && e.maxColour < 0.005, "\(e)")
    }

    @Test("straight alpha is premultiplied first")
    func straightAlpha() throws {
        var picture = Self.coloured
        // Make it straight: un-premultiply in place.
        for i in 0..<(picture.width * picture.height) where picture.rgba[4 * i + 3] > 0 {
            let a = Float(picture.rgba[4 * i + 3])
            for c in 0..<3 { picture.rgba[4 * i + c] = UInt8(min(255, (Float(picture.rgba[4 * i + c]) * 255 / a).rounded())) }
        }
        let upscaler = try SubtitleUpscaler(submission: Metal3Submission.shared)
        let e = Self.error(try Self.run(picture, upscaler, straight: true), Self.reference(picture, configuration: .default, straight: true))
        #expect(e.maxAlpha < 0.005 && e.maxColour < 0.005, "\(e)")
    }

    @Test("CGImage, pixel buffer and arbitrary-size conveniences")
    func conveniences() throws {
        let upscaler = try SubtitleUpscaler(submission: Metal3Submission.shared)
        let picture = SubtitleUpscaleFixtures.glyphs(width: 120, height: 40, lines: ["DVD"], fontSize: 30)
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let provider = CGDataProvider(data: Data(picture.rgba) as CFData)!
        let image = CGImage(width: 120, height: 40, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 480, space: space,
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider,
                            decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let twice = try upscaler.upscale(image)
        #expect(twice.width == 240 && twice.height == 80)
        // The 8-bit result is the Float32 result (on the same content) rounded.
        let exact = try Self.run(picture, upscaler, content: Self.content(picture))
        let bytes = Array(twice.dataProvider!.data! as Data)
        var worst = 0
        for (i, p) in exact.pixels.enumerated() {
            for c in 0..<4 { worst = max(worst, abs(Int(bytes[4 * i + c]) - Int((p[c] * 255).rounded()))) }
        }
        #expect(worst <= 1)
        // DVD-style 4.5x: two 2x passes, then Lanczos to the exact size.
        let large = try upscaler.upscale(image, size: CGSize(width: 540, height: 180))
        #expect(large.width == 540 && large.height == 180)
        let largeBytes = Array(large.dataProvider!.data! as Data)
        for i in stride(from: 0, to: largeBytes.count, by: 4) {
            #expect(largeBytes[i] <= largeBytes[i + 3] && largeBytes[i + 1] <= largeBytes[i + 3] && largeBytes[i + 2] <= largeBytes[i + 3])
        }
        let pixelBuffer = try upscaler.upscale(try Self.pixelBuffer(picture), size: CGSize(width: 300, height: 100))
        #expect(CVPixelBufferGetWidth(pixelBuffer) == 300 && CVPixelBufferGetHeight(pixelBuffer) == 100)
    }

    static func pixelBuffer(_ p: SubtitleUpscaleFixtures.Picture) throws -> CVPixelBuffer {
        var b: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any], kCVPixelBufferMetalCompatibilityKey: true]
        CVPixelBufferCreate(nil, p.width, p.height, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &b)
        let buffer = try #require(b)
        CVPixelBufferLockBaseAddress(buffer, [])
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<p.height {
            for x in 0..<p.width {
                let s = (y * p.width + x) * 4
                base[y * stride + 4 * x] = p.rgba[s + 2]
                base[y * stride + 4 * x + 1] = p.rgba[s + 1]
                base[y * stride + 4 * x + 2] = p.rgba[s]
                base[y * stride + 4 * x + 3] = p.rgba[s + 3]
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }
}
