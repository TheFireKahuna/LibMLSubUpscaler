import Metal
import Testing
@_spi(Archive) @testable import LibMLSubUpscaler

struct PipelineTests {
    /// The archive carries the pipelines the upscaler makes: one request per kernel, each in the precompiled library.
    @Test("the archive's requests are the upscaler's kernels, each in the precompiled library")
    func archiveRequests() throws {
        let library = try SubtitleUpscaleKernels.library(for: Metal3Submission.shared.device)
        let requests = SubtitleUpscaler.archiveRequests(library: library)
        #expect(requests.map(\.function) == SubtitleUpscaleKernels.functions)
        for request in requests { #expect(library.makeFunction(name: request.function) != nil, "\(request.function)") }
    }
}
