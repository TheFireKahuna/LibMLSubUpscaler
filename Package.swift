// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "LibMLSubUpscaler",
    platforms: [
        .iOS(.v18),
        .tvOS(.v18),
        .macOS(.v15),
        .visionOS(.v2),
    ],
    products: [
        .library(name: "LibMLSubUpscaler", targets: ["LibMLSubUpscaler"]),
    ],
    targets: [
        .target(
            name: "LibMLSubUpscaler",
            // The compiled ArtCNN models, copied as they are: `.process` would have Xcode recompile them and
            // generate classes, which `swift build` does not.
            resources: [.copy("Models")],
            linkerSettings: [
                .linkedFramework("CoreML"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("Metal"),
            ]
        ),
        .testTarget(
            name: "LibMLSubUpscalerTests",
            dependencies: ["LibMLSubUpscaler"],
            resources: [.copy("Weights")]
        ),
    ]
)
