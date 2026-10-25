import Foundation

/// The ArtCNN line-art models ``SubtitleUpscaler`` runs. Both are the neutral 2x luma models (not the `_DS`
/// sharpening or `_DN` denoising variants); C4F32 has twice the channels and four times the work of C4F16.
public enum ArtCNNModel: String, Sendable, CaseIterable {
    case c4f16
    case c4f32
}
