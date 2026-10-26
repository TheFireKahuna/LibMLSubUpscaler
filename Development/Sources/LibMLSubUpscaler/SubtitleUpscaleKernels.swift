import Foundation
import Metal

/// The process's one set of upscaler GPU state: device and the pipelines of the passes around the Neural Engine, made by
/// the first submission asked (``SubtitleUpscaler/prewarm(_:submission:)`` asks off the caller's thread) and shared by
/// every ``SubtitleUpscaler`` on its device.
final class SubtitleUpscaleKernels: @unchecked Sendable {
    enum KernelError: Error, CustomStringConvertible {
        case unavailable(String)
        var description: String {
            switch self { case .unavailable(let why): "subtitle upscaler unavailable: \(why)" }
        }
    }

    let device: MTLDevice
    let maskBits: MTLComputePipelineState
    let maskRows: MTLComputePipelineState
    let maskColumns: MTLComputePipelineState
    let tileGather: MTLComputePipelineState
    let compositeTiles: MTLComputePipelineState
    let clearRect: MTLComputePipelineState
    let resample: MTLComputePipelineState

    private static let sharedLock = NSLock()
    nonisolated(unsafe) private static var shared: Result<SubtitleUpscaleKernels, Error>?

    /// The shared kernels, made by `submission` on the first call (which blocks for it); a failure is kept, so a device
    /// without them costs one attempt.
    static func get(_ submission: any SubtitleUpscaleSubmission) throws -> SubtitleUpscaleKernels {
        sharedLock.lock()
        defer { sharedLock.unlock() }
        if let shared { return try shared.get() }
        let made = Result { try SubtitleUpscaleKernels(submission: submission) }
        shared = made
        return try made.get()
    }

    /// The upscaler's kernels, precompiled into the module's bundle (Shaders/LibMLSubUpscaler.metal).
    static func library(for device: MTLDevice) throws -> MTLLibrary { try device.makeDefaultLibrary(bundle: .module) }

    /// The pipelines translated ahead of time for the GPUs the bundle was built for (a tvOS device's: the A15's), which
    /// a Metal 4 compiler takes a pipeline from instead of compiling it; a bundle built for another platform carries
    /// none (Scripts/build-xcframework.sh).
    static let archive = Bundle.module.url(forResource: "LibMLSubUpscaler-pipelines", withExtension: "metallib")

    /// Every kernel, in the order the passes run them.
    static let functions = ["up_mask_bits", "up_mask_rows", "up_mask_columns", "up_tile_gather", "up_composite_tiles",
                            "up_clear_rect", "up_resample"]


    /// A fresh set made by `submission`: ``get(_:)`` makes the shared one; the benchmark makes its own to time it.
    init(submission: any SubtitleUpscaleSubmission) throws {
        device = submission.device
        let library = try Self.library(for: device)
        func make(_ name: String) throws -> MTLComputePipelineState {
            try submission.makeComputePipeline(SubtitleUpscalePipelineRequest(library: library, function: name,
                                                                              archive: Self.archive))
        }
        maskBits = try make("up_mask_bits")
        maskRows = try make("up_mask_rows")
        maskColumns = try make("up_mask_columns")
        tileGather = try make("up_tile_gather")
        compositeTiles = try make("up_composite_tiles")
        clearRect = try make("up_clear_rect")
        resample = try make("up_resample")
    }
}

extension SubtitleUpscaler {
    /// Every pipeline the upscaler makes, as requests on `library` (a tvOS build's default.metallib, loaded on the Mac):
    /// what the build records in the pipeline script it translates for the TV's GPU.
    @_spi(Archive) public static func archiveRequests(library: MTLLibrary) -> [SubtitleUpscalePipelineRequest] {
        SubtitleUpscaleKernels.functions.map { SubtitleUpscalePipelineRequest(library: library, function: $0, archive: nil) }
    }
}
