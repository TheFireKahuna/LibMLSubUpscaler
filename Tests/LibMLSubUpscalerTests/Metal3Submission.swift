import Foundation
import Metal
import LibMLSubUpscaler

/// The tests' submission: a Metal 3 queue, a command buffer a batch and a compute encoder a pass, events encoded into
/// the command buffer. A player brings its own (its Metal 4 or Metal 3 queue); this is the plainest one that runs the
/// upscaler.
final class Metal3Submission: SubtitleUpscaleSubmission, @unchecked Sendable {
    /// On the system's default device, the upscaler's.
    static let shared = Metal3Submission(device: MTLCreateSystemDefaultDevice()!)!

    let device: MTLDevice
    private let queue: MTLCommandQueue

    init?(device: MTLDevice) {
        guard let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.queue = queue
    }

    func makeBatch() -> (any SubtitleUpscaleBatch)? {
        queue.makeCommandBuffer().map(Batch.init)
    }

    private final class Batch: SubtitleUpscaleBatch {
        let commandBuffer: MTLCommandBuffer
        var open: Pass?
        var signals: [(MTLEvent, UInt64)] = []

        init(_ commandBuffer: MTLCommandBuffer) { self.commandBuffer = commandBuffer }

        func wait(for event: MTLEvent, value: UInt64) { commandBuffer.encodeWaitForEvent(event, value: value) }
        func signal(_ event: MTLEvent, value: UInt64) { signals.append((event, value)) }

        func makePass() -> (any SubtitleUpscalePass)? {
            open?.encoder.endEncoding()
            open = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent).map(Pass.init)
            return open
        }

        func commit(_ completion: @escaping @Sendable (String?) -> Void) {
            open?.encoder.endEncoding()
            open = nil
            for (event, value) in signals { commandBuffer.encodeSignalEvent(event, value: value) }
            commandBuffer.addCompletedHandler { completion($0.error.map { "\($0)" }) }
            commandBuffer.commit()
        }
    }

    private final class Pass: SubtitleUpscalePass {
        let encoder: MTLComputeCommandEncoder

        init(_ encoder: MTLComputeCommandEncoder) { self.encoder = encoder }

        func setPipeline(_ pipeline: MTLComputePipelineState) { encoder.setComputePipelineState(pipeline) }
        func setTexture(_ texture: MTLTexture, index: Int) { encoder.setTexture(texture, index: index) }
        func setBuffer(_ buffer: MTLBuffer, offset: Int, index: Int) { encoder.setBuffer(buffer, offset: offset, index: index) }
        func setBytes(_ bytes: UnsafeRawBufferPointer, index: Int) {
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: index)
        }
        func dispatchThreads(_ grid: MTLSize, threadsPerThreadgroup: MTLSize) {
            encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        func barrier() { encoder.memoryBarrier(scope: [.buffers, .textures]) }
    }
}
