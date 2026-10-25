import Testing
import Foundation
@testable import LibMLSubUpscaler

/// GPU time of the subtitle upscaler per picture (``SubtitleUpscaler/benchmark(iterations:sweep:)``, the same
/// numbers a probe app gets on a device). Not a test: it asserts nothing and only runs with `SUBTITLE_UPSCALE_BENCH=1`.
@Suite(.serialized)
struct SubtitleUpscaleBenchmarkTests {
    static let enabled = ProcessInfo.processInfo.environment["SUBTITLE_UPSCALE_BENCH"] == "1"

    @Test("app-open, first-picture and steady-state costs", .enabled(if: enabled))
    func benchmark() {
        print(SubtitleUpscaler.benchmark())
    }
}
