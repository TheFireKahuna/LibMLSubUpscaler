import Testing
import Foundation
@testable import LibMLSubUpscaler

struct SubtitleUpscaleTilePlanTests {
    typealias Content = SubtitleUpscaler.Content

    /// Content of lines (y, height, first column, last column) in a picture.
    static func lines(_ lines: [(Int, Int, Int, Int)]) -> Content {
        let y0 = lines.map(\.0).min()!, y1 = lines.map { $0.0 + $0.1 }.max()!
        var spans = [ClosedRange<Int>?](repeating: nil, count: y1 - y0)
        for (y, h, a, b) in lines { for r in y..<(y + h) { spans[r - y0] = a...b } }
        let x0 = lines.map(\.2).min()!, x1 = lines.map(\.3).max()!
        return Content(bounds: CGRect(x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0), spans: spans)
    }

    static let pictures: [(name: String, width: Int, height: Int, content: Content)] = [
        ("a pixel", 1, 1, Content(bounds: CGRect(x: 0, y: 0, width: 1, height: 1))),
        ("a strip", 1400, 120, lines([(20, 40, 100, 1290), (70, 40, 300, 1100)])),
        ("uneven lines", 1500, 150, lines([(10, 56, 0, 1499), (90, 50, 600, 900)])),
        ("lines far apart", 1920, 1080, lines([(40, 50, 700, 1200), (950, 60, 200, 1700)])),
        ("a wide rectangle", 3840, 300, Content(bounds: CGRect(x: 0, y: 0, width: 3840, height: 300))),
        ("a canvas", 1920, 1080, Content(bounds: CGRect(x: 0, y: 0, width: 1920, height: 1080))),
    ]

    static func plan(_ p: (name: String, width: Int, height: Int, content: Content), alpha: ArtCNNModel = .c4f16, luma: ArtCNNModel = .c4f32) -> SubtitleUpscaleTilePlan {
        SubtitleUpscaleTilePlan.plan(content: p.content, width: p.width, height: p.height, support: 5, alpha: alpha, luma: luma)
    }

    @Test("each window is its core plus the reach, every shape a model's, the cores of a band side by side",
          arguments: pictures.map(\.name))
    func windows(_ name: String) {
        let p = Self.pictures.first { $0.name == name }!
        let plan = Self.plan(p)
        var k = 0
        for band in plan.bands {
            var x = band.x
            for c in band.columns {
                let w = plan.windows[k]
                #expect(w == .init(x: x - 7, y: band.y - 7, width: c + 14, height: band.height + 14))
                #expect(ArtCNNCoreMLModels.windowHeights.contains(w.height) && ArtCNNCoreMLModels.windowWidths.contains(w.width))
                x += c; k += 1
            }
        }
        #expect(k == plan.windows.count)
        // Bands in order, none overlapping.
        for (a, b) in zip(plan.bands, plan.bands.dropFirst()) { #expect(a.y + a.height <= b.y) }
    }

    @Test("the cores cover every pixel within the support of content, and the cores and the cleared rectangles cover each pixel once",
          arguments: pictures.map(\.name))
    func coverage(_ name: String) {
        let p = Self.pictures.first { $0.name == name }!
        let plan = Self.plan(p)
        let kept = p.content.kept(width: p.width, height: p.height, support: 5)
        var count = [UInt8](repeating: 0, count: p.width * p.height)
        for w in plan.windows {
            for y in max(w.y + 7, kept.y)..<min(w.y + w.height - 7, kept.y + kept.height) {
                for x in max(w.x + 7, kept.x)..<min(w.x + w.width - 7, kept.x + kept.width) { count[y * p.width + x] += 1 }
            }
        }
        var supported = [Bool](repeating: false, count: p.width * p.height)
        for y in 0..<p.height {
            guard let s = p.content.span(row: y) else { continue }
            for yy in max(0, y - 5)...min(p.height - 1, y + 5) {
                for x in max(0, s.lowerBound - 5)...min(p.width - 1, s.upperBound + 5) { supported[yy * p.width + x] = true }
            }
        }
        for i in 0..<count.count where supported[i] { #expect(count[i] == 1, "\(name): supported pixel \(i % p.width),\(i / p.width) in \(count[i]) cores") }
        for r in SubtitleUpscaler.uncovered(plan, kept: kept, width: p.width, height: p.height) {
            for y in r.y..<(r.y + r.height) { for x in r.x..<(r.x + r.width) { count[y * p.width + x] += 1 } }
        }
        #expect(count.allSatisfy { $0 == 1 }, "\(name): \(count.filter { $0 != 1 }.count) pixels covered other than once")
    }

    @Test("no content, no windows")
    func empty() {
        #expect(SubtitleUpscaleTilePlan.plan(content: .empty, width: 100, height: 50, support: 5, alpha: .c4f16, luma: .c4f16).windows.isEmpty)
    }

    @Test("lines of different lengths, and lines far apart, cost less planned per band than as their rectangle")
    func perBand() {
        func cost(_ plan: SubtitleUpscaleTilePlan) -> Double {
            plan.windows.reduce(0) { $0 + SubtitleUpscaleTilePlan.milliseconds(alpha: .c4f16, luma: .c4f32, height: $1.height, width: $1.width) }
        }
        for name in ["uneven lines", "lines far apart"] {
            let p = Self.pictures.first { $0.name == name }!
            let rows = cost(Self.plan(p))
            let rectangle = cost(Self.plan((p.name, p.width, p.height, Content(bounds: p.content.bounds))))
            #expect(rows < 0.9 * rectangle, "\(name): \(rows) ms per band against \(rectangle) ms as a rectangle")
        }
    }
}
