import Foundation
import Metal

/// The caller's GPU submission, which ``SubtitleUpscaler`` encodes into. The upscaler owns no command queue: its GPU
/// passes run on the caller's queue with the caller's Metal (a Metal 4 argument table or a Metal 3 encoder behind
/// the same calls).
public protocol SubtitleUpscaleSubmission: AnyObject, Sendable {
    var device: MTLDevice { get }
    /// The compute pipeline `request` describes, made by the caller's Metal: on Metal 4, its compiler, looking in the
    /// request's archive first; before it, the device. The upscaler calls this once per pipeline, for the process.
    func makeComputePipeline(_ request: SubtitleUpscalePipelineRequest) throws -> MTLComputePipelineState
    /// A new batch, which starts on the GPU only after the work committed before it from this submission.
    func makeBatch() -> (any SubtitleUpscaleBatch)?
}

/// A pipeline the upscaler needs: `function` of its precompiled `library`. `archive` is the upscaler's binary archive for
/// this build's GPUs (a Metal 4 archive the build translated ahead of time; nil where the bundle carries none), which a
/// Metal 4 compiler can take the pipeline from instead of compiling it.
public struct SubtitleUpscalePipelineRequest {
    public let library: MTLLibrary
    public let function: String
    public let archive: URL?
}

/// Work for one commit, encoded by one caller and committed once.
public protocol SubtitleUpscaleBatch: AnyObject {
    /// Before the first pass: the batch's work waits on the GPU until `event` reaches `value`.
    func wait(for event: MTLEvent, value: UInt64)
    /// `event` set to `value` once the batch's work has completed on the GPU.
    func signal(_ event: MTLEvent, value: UInt64)
    /// A compute pass in this batch; the previous one, if still open, is ended first.
    func makePass() -> (any SubtitleUpscalePass)?
    /// Ends the open pass and commits the batch. `completion` runs once, off the caller's thread, with nil when the
    /// batch completed or why it failed.
    func commit(_ completion: @escaping @Sendable (_ failure: String?) -> Void)
}

/// A compute pass. Dispatches in a pass may run together unless a ``barrier()`` separates them; bindings are by index,
/// as the kernels declare them.
public protocol SubtitleUpscalePass: AnyObject {
    func setPipeline(_ pipeline: MTLComputePipelineState)
    func setTexture(_ texture: MTLTexture, index: Int)
    func setBuffer(_ buffer: MTLBuffer, offset: Int, index: Int)
    /// `bytes` copied into the batch and bound at `index`, as a small constant argument.
    func setBytes(_ bytes: UnsafeRawBufferPointer, index: Int)
    func dispatchThreads(_ grid: MTLSize, threadsPerThreadgroup: MTLSize)
    /// Every dispatch after this sees every write before it.
    func barrier()
}

extension SubtitleUpscalePass {
    func setValue<T>(_ value: T, index: Int) {
        withUnsafeBytes(of: value) { setBytes($0, index: index) }
    }
}
