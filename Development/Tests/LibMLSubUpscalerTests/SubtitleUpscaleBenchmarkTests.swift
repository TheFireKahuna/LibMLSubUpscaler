import Testing
import Foundation
@testable import LibMLSubUpscaler

/// The subtitle upscaler's costs (``SubtitleUpscaler/benchmark(iterations:submission:)``, the same numbers a probe app gets on
/// a device). Not a test: it asserts nothing and only runs with `SUBTITLE_UPSCALE_BENCH=1`.
@Suite(.serialized)
struct SubtitleUpscaleBenchmarkTests {
    static let enabled = ProcessInfo.processInfo.environment["SUBTITLE_UPSCALE_BENCH"] == "1"

    @Test("app-open, first-picture and steady-state costs", .enabled(if: enabled))
    func benchmark() {
        print(SubtitleUpscaler.benchmark(submission: Metal3Submission.shared))
    }

    @Test("cold Neural Engine loads, serial and parallel", .enabled(if: enabled))
    func coldLoads() {
        print(SubtitleUpscaleNeuralEngine.coldLoadReport())
    }
}
