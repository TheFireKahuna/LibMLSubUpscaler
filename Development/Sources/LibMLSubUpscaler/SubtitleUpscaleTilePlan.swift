import Foundation

/// How the upscaler covers a picture with the models' fixed windows (``ArtCNNCoreMLModels/windowHeights`` x
/// ``ArtCNNCoreMLModels/windowWidths``). Each window is a core plus the network's 7-pixel reach on every side, so a
/// core's outputs are exact.
///
/// Only the support needs computing: the pixels within S (the support radius) of content; everything else is exactly
/// transparent. The plan cuts the rows into bands, each band one core height tall, and covers each band with a row of
/// cores across just the columns its own support spans, so two lines of different lengths, or the gap between them,
/// cost only what they hold. Rows with no support take no band. The choice is a dynamic programme over the rows (a
/// band of each height, or skipping an unsupported row), each band's columns the cheapest cover of its span, every
/// window costed by its measured prediction time (``milliseconds(alpha:luma:height:width:)``).
struct SubtitleUpscaleTilePlan: Equatable {
    struct Window: Equatable {
        /// Picture position of the window's top-left pixel: its core's minus 7 on each axis.
        var x: Int
        var y: Int
        var width: Int
        var height: Int
    }

    /// A band's rows (core height) from `y`, its cores' widths from column `x`, in picture coordinates.
    struct Band: Equatable {
        var y: Int
        var height: Int
        var x: Int
        var columns: [Int]
        var width: Int { columns.reduce(0, +) }
    }

    var bands: [Band]
    /// Band by band, left to right.
    var windows: [Window]

    static let reach = 7

    /// Wall milliseconds of one synchronous prediction of both of a window's planes (`alpha` and `luma` the
    /// networks on them), measured on the A15 (Apple TV 4K, tvOS 27.2, Release, the `fastPrediction`
    /// specialisation; ``SubtitleUpscaler/benchmark(iterations:submission:)`` prints the device's own). Proportions, not the
    /// absolute times, steer the plan.
    static func milliseconds(alpha: ArtCNNModel, luma: ArtCNNModel, height: Int, width: Int) -> Double {
        let table: [Int: [Int: Double]] = switch (alpha, luma) {
        case (.c4f16, .c4f16): [46: [256: 1.09, 512: 1.27, 1024: 1.30, 2048: 1.73],
                                  78: [256: 0.54, 512: 0.75, 1024: 1.17, 2048: 2.14],
                                  142: [256: 0.68, 512: 1.05, 1024: 1.84, 2048: 3.65]]
        case (.c4f16, .c4f32): [46: [256: 0.57, 512: 0.76, 1024: 1.15, 2048: 2.09],
                                  78: [256: 0.68, 512: 0.96, 1024: 1.67, 2048: 3.72],
                                  142: [256: 0.86, 512: 1.47, 1024: 2.76, 2048: 6.29]]
        case (.c4f32, .c4f16): [46: [256: 0.56, 512: 0.72, 1024: 1.05, 2048: 1.91],
                                  78: [256: 0.66, 512: 0.96, 1024: 1.59, 2048: 2.97],
                                  142: [256: 0.86, 512: 1.49, 1024: 2.54, 2048: 5.62]]
        case (.c4f32, .c4f32): [46: [256: 0.68, 512: 0.83, 1024: 1.46, 2048: 2.75],
                                  78: [256: 0.73, 512: 1.18, 1024: 2.19, 2048: 4.94],
                                  142: [256: 1.05, 512: 1.89, 1024: 3.95, 2048: 9.07]]
        }
        return table[height]![width]!
    }

    /// The plan for `content` in a `width` x `height` picture, keeping pixels within `support` of it, with `alpha`
    /// and `luma` the networks on the two planes.
    static func plan(content: SubtitleUpscaler.Content, width: Int, height: Int, support: Int,
                     alpha: ArtCNNModel, luma: ArtCNNModel) -> Self {
        let r = reach
        let coreHeights = ArtCNNCoreMLModels.windowHeights.map { $0 - 2 * r }
        let coreWidths = ArtCNNCoreMLModels.windowWidths.map { $0 - 2 * r }
        let kept = content.kept(width: width, height: height, support: support)
        guard kept.width > 0 else { return Self(bands: [], windows: []) }
        let y0 = kept.y, y1 = kept.y + kept.height
        // The columns each picture row's support spans: its own content and that of the rows within `support`,
        // each grown by `support` and clipped to the picture.
        var span = [(Int, Int)?](repeating: nil, count: height)
        for y in max(0, y0 - support)..<min(height, y1 + support) {
            guard let s = content.span(row: y) else { continue }
            let lo = max(0, s.lowerBound - support), hi = min(width - 1, s.upperBound + support)
            for yy in max(0, y - support)...min(height - 1, y + support) {
                span[yy] = span[yy].map { (min($0.0, lo), max($0.1, hi)) } ?? (lo, hi)
            }
        }
        // The cheapest cover of `length` columns with cores of height `h`: per height one table up to the widest span
        // (the kept width), each entry its cost and its last core, followed back to list the cores.
        var tables: [Int: (cost: [Double], last: [Int])] = [:]
        func table(_ h: Int) -> (cost: [Double], last: [Int]) {
            if let made = tables[h] { return made }
            let n = kept.width
            let prices = coreWidths.map { milliseconds(alpha: alpha, luma: luma, height: h + 2 * r, width: $0 + 2 * r) }
            var cost = [Double](repeating: .infinity, count: n + 1)
            var last = [Int](repeating: 0, count: n + 1)
            cost[0] = 0
            for t in 1...n {
                for (k, c) in coreWidths.enumerated() {
                    let v = cost[max(0, t - c)] + prices[k]
                    if v < cost[t] { cost[t] = v; last[t] = c }
                }
            }
            tables[h] = (cost, last)
            return (cost, last)
        }
        func cover(_ length: Int, _ h: Int) -> (cost: Double, cores: [Int]) {
            let t = table(h)
            var cores: [Int] = []
            var at = length
            while at > 0 { cores.append(t.last[at]); at -= t.last[at] }
            return (t.cost[length], cores)
        }
        func coverCost(_ length: Int, _ h: Int) -> Double { table(h).cost[length] }
        // A band of core height h from row t: the union of its rows' spans, nil when it holds no support.
        func bandSpan(_ t: Int, _ h: Int) -> (Int, Int)? {
            var lo = Int.max, hi = -1
            for y in t..<min(y1, t + h) { if let s = span[y] { lo = min(lo, s.0); hi = max(hi, s.1) } }
            return hi < 0 ? nil : (lo, hi)
        }
        // best[i]: the cheapest plan for rows y0 + i ..< y1, and its first step (0 skips a row).
        let n = y1 - y0
        var best = [Double](repeating: .infinity, count: n + 1)
        var step = [Int](repeating: 0, count: n + 1)
        best[n] = 0
        for i in stride(from: n - 1, through: 0, by: -1) {
            let t = y0 + i
            if span[t] == nil { best[i] = best[i + 1]; step[i] = 0 }
            for h in coreHeights {
                let rest = best[min(n, i + h)]
                guard let s = bandSpan(t, h) else { continue }
                let cost = coverCost(s.1 - s.0 + 1, h) + rest
                if cost < best[i] { best[i] = cost; step[i] = h }
            }
        }
        var bands: [Band] = []
        var windows: [Window] = []
        var i = 0
        while i < n {
            let h = step[i]
            guard h > 0 else { i += 1; continue }
            let t = y0 + i
            let s = bandSpan(t, h)!
            let cores = cover(s.1 - s.0 + 1, h).cores.sorted(by: >)
            bands.append(Band(y: t, height: h, x: s.0, columns: cores))
            var x = s.0
            for c in cores {
                windows.append(Window(x: x - r, y: t - r, width: c + 2 * r, height: h + 2 * r))
                x += c
            }
            i += h
        }
        return Self(bands: bands, windows: windows)
    }
}
