// swift-tools-version: 6.1

import PackageDescription

// The upscaler's source, tests and build. Players depend on the package at the repository's root, which ships the
// XCFramework Scripts/build-xcframework.sh makes from this one.
let package = Package(
    name: "LibMLSubUpscaler",
    platforms: [
        .iOS(.v18),
        .tvOS(.v18),
        .macOS(.v15),
        .visionOS(.v2),
    ],
    products: [
        .library(name: "LibMLSubUpscaler", type: .dynamic, targets: ["LibMLSubUpscaler"]),
    ],
    targets: [
        .target(
            name: "LibMLSubUpscaler",
            // The compiled ArtCNN models, copied as they are: `.process` would have Xcode recompile them and
            // generate classes, which `swift build` does not.
            // Shaders: compiled into the bundle's default.metallib when the package builds.
            resources: [.copy("Models"), .process("Shaders")],
            linkerSettings: [
                .linkedFramework("CoreML"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("Metal"),
            ]
        ),
        // The build's tool: the upscaler's pipelines, in a pipeline script for metal-tt (Scripts/build-xcframework.sh).
        .executableTarget(name: "Harvest", dependencies: ["LibMLSubUpscaler"], path: "Sources/Harvest"),
        .testTarget(
            name: "LibMLSubUpscalerTests",
            dependencies: ["LibMLSubUpscaler"],
            resources: [.copy("Weights")]
        ),
    ]
)
