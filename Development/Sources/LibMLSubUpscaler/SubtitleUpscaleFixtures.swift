import CoreGraphics
import CoreText
import Foundation
import Metal

/// Synthetic subtitle bitmaps for the upscaler's tests and benchmark: anti-aliased glyphs in a fill colour with a
/// dark outline on transparency, as a PGS or DVB display set renders, premultiplied RGBA8 (R, G, B, A bytes).
enum SubtitleUpscaleFixtures {
    struct Picture {
        let width: Int
        let height: Int
        /// Premultiplied R, G, B, A per pixel, row 0 at the top.
        var rgba: [UInt8]
    }

    /// `lines` of text centred in a `width` x `height` canvas whose lower part they occupy (`bottomFraction` of the
    /// height from the bottom edge for the last line's baseline), filled with `fill` and stroked dark.
    static func glyphs(width: Int, height: Int, lines: [String] = ["The quick brown fox jumps over the lazy dog."],
                       fontSize: CGFloat = 46, fill: (CGFloat, CGFloat, CGFloat) = (1, 1, 1),
                       outline: CGFloat = 3, baselineFromBottom: CGFloat? = nil) -> Picture {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let info = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        rgba.withUnsafeMutableBytes { raw in
            guard let context = CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width * 4, space: space, bitmapInfo: info) else { return }
            context.setAllowsAntialiasing(true)
            context.setShouldSmoothFonts(false)
            let font = CTFontCreateWithName("Helvetica-Bold" as CFString, fontSize, nil)
            let lineHeight = fontSize * 1.25
            var baseline = baselineFromBottom ?? max((CGFloat(height) - lineHeight * CGFloat(lines.count)) / 2, 0) + fontSize * 0.3
            for text in lines.reversed() {
                for pass in 0..<2 {
                    // The outline first (stroked twice as wide, the inner half then covered by the fill), as a
                    // subtitle renderer draws it.
                    let attributes: [CFString: Any] = [
                        kCTFontAttributeName: font,
                        kCTForegroundColorAttributeName: CGColor(srgbRed: fill.0, green: fill.1, blue: fill.2, alpha: 1),
                        kCTStrokeColorAttributeName: CGColor(srgbRed: 0.05, green: 0.05, blue: 0.05, alpha: 1),
                        kCTStrokeWidthAttributeName: pass == 0 ? NSNumber(value: Double(outline * 2 / fontSize * 100)) : NSNumber(value: 0),
                    ]
                    let line = CTLineCreateWithAttributedString(CFAttributedStringCreate(nil, text as CFString, attributes as CFDictionary))
                    let bounds = CTLineGetBoundsWithOptions(line, [])
                    context.textPosition = CGPoint(x: (CGFloat(width) - bounds.width) / 2, y: baseline)
                    CTLineDraw(line, context)
                }
                baseline += lineHeight
            }
        }
        return Picture(width: width, height: height, rgba: rgba)
    }

    /// A shared RGBA8 texture holding `picture`.
    static func texture(_ picture: Picture, device: MTLDevice) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: picture.width, height: picture.height, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else { return nil }
        picture.rgba.withUnsafeBytes {
            t.replace(region: MTLRegionMake2D(0, 0, picture.width, picture.height), mipmapLevel: 0, withBytes: $0.baseAddress!,
                      bytesPerRow: picture.width * 4)
        }
        return t
    }
}
