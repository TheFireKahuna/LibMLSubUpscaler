import Foundation

/// ArtCNN as compiled Core ML models (`Models/ArtCNN_<alpha>_<luma>.mlmodelc`, built by `Scripts/ArtCNN` from ArtCNN
/// commit 0837e5326e710f49e2ab0a8d995da0708047442e, MIT, see ``ArtCNNLicense``) for ``SubtitleUpscaleNeuralEngine``,
/// one per pairing of an alpha and a luma network, both planes of a window in one call: function `h<H>w<W>` takes `xa`
/// and `xl`, Float16 `(1, 1, H, W)`, and gives `ya` and `yl`, Float16 `(1, 4, H, W)`, each network's last
/// convolution's four sub-pixel channels (channel `2 * row + column` of each 2x block), unclamped. Fixed shapes
/// because Core ML runs a flexible shape on the Neural Engine only at its default size, or re-optimises it for
/// milliseconds at each change; the functions of one model share its weights and switch for free.
enum ArtCNNCoreMLModels {
    /// Window heights and widths, each a core plus the network's 7-pixel reach on both sides; the widths keep
    /// every row a multiple of 64 bytes.
    static let windowHeights = [46, 78, 142]
    static let windowWidths = [256, 512, 1024, 2048]

    /// The compiled model of the `alpha` / `luma` pairing in the package's resources.
    static func url(alpha: ArtCNNModel, luma: ArtCNNModel) throws -> URL {
        let name = "ArtCNN_\(alpha.rawValue.uppercased())_\(luma.rawValue.uppercased())"
        guard let url = Bundle.module.url(forResource: name, withExtension: "mlmodelc", subdirectory: "Models") else {
            throw SubtitleUpscaler.UpscaleError.unavailable("\(name).mlmodelc is not in the package's resources")
        }
        return url
    }
}
