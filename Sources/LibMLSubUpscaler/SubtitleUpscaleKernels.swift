import Foundation
import Metal

/// The process's one set of upscaler GPU state: device, queue and the compiled pipelines of the passes around the
/// Neural Engine. Built once, on first use (``SubtitleUpscaler/prewarm()`` builds it off the caller's thread), and
/// shared by every ``SubtitleUpscaler``.
final class SubtitleUpscaleKernels: @unchecked Sendable {
    enum KernelError: Error, CustomStringConvertible {
        case unavailable(String)
        var description: String {
            switch self { case .unavailable(let why): "subtitle upscaler unavailable: \(why)" }
        }
    }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let maskBits: MTLComputePipelineState
    let maskRows: MTLComputePipelineState
    let maskColumns: MTLComputePipelineState
    let tileGather: MTLComputePipelineState
    let compositeTiles: MTLComputePipelineState
    let clearRect: MTLComputePipelineState
    let resample: MTLComputePipelineState

    private static let sharedLock = NSLock()
    nonisolated(unsafe) private static var shared: Result<SubtitleUpscaleKernels, Error>?

    /// The shared kernels, compiled on the first call (which blocks for the compile); a failure is kept, so a
    /// device without Metal costs one attempt.
    static func get() throws -> SubtitleUpscaleKernels {
        sharedLock.lock()
        defer { sharedLock.unlock() }
        if let shared { return try shared.get() }
        let made = Result { try SubtitleUpscaleKernels() }
        shared = made
        return try made.get()
    }

    /// A fresh set: ``get()`` makes the shared one; the benchmark makes its own to time the compile.
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw KernelError.unavailable("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw KernelError.unavailable("no command queue") }
        queue.label = "SubtitleUpscaler"
        self.device = device
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .fast
        let library = try device.makeLibrary(source: SubtitleUpscaleShaderSource.source, options: options)
        func make(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw KernelError.unavailable("no kernel \(name)") }
            return try device.makeComputePipelineState(function: f)
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
