// swift-tools-version: 6.1

import PackageDescription

// LibMLSubUpscaler as players take it: the XCFramework Development/Scripts/build-xcframework.sh builds from the source
// package in Development/ (where the tests run). Library evolution gives it a stable ABI; its bundle carries the
// precompiled shaders and the ArtCNN models.
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
        .binaryTarget(name: "LibMLSubUpscaler", path: "Frameworks/LibMLSubUpscaler.xcframework"),
    ]
)
