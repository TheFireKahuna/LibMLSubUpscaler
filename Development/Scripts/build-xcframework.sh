#!/bin/zsh
# build-xcframework.sh: Frameworks/LibMLSubUpscaler.xcframework at the repository's root, from this package.
#
# Every platform's framework, built with library evolution (a stable ABI and a .swiftinterface), carries its precompiled
# shaders and the ArtCNN models (the module's bundle: default.metallib, Models/). The tvOS device slice also carries a
# Metal 4 binary archive of every pipeline the upscaler makes, translated ahead of time
# for the A15 (Apple TV 4K, 3rd generation): its compiler takes those pipelines from the archive instead of compiling
# them. Metal takes an archive only on the OS release it was built for (TVOS_ARCHIVE_OS); on any other release, and on
# GPUs it was not built for, the pipelines are compiled when the player prewarms the upscaler.
#
# Needs Xcode 27.2 (DEVELOPER_DIR=~/Downloads/Xcode-beta.app/Contents/Developer: metal-tt) on macOS 26 or later.
set -euo pipefail

HERE=${0:A:h}
DEV=${HERE:h}
ROOT=${DEV:h}
NAME=LibMLSubUpscaler
DD=${DD:-$DEV/.build/xcframework}
TVOS_ARCHIVE_OS=${TVOS_ARCHIVE_OS:-27.0}
ARCHIVE_GPUS=(applegpu_g14p)

# destination : products directory suffix
SLICES=(
  "tvOS:-appletvos"
  "tvOS Simulator:-appletvsimulator"
  "iOS:-iphoneos"
  "iOS Simulator:-iphonesimulator"
  "macOS:"
  "visionOS:-xros"
  "visionOS Simulator:-xrsimulator"
)

cd $DEV
mkdir -p $DD/logs
for slice in $SLICES; do
  destination=${slice%%:*}
  log=$DD/logs/${destination// /-}.log
  print "== $destination"
  # The visionOS simulator is arm64 alone, as the other libraries' slices are.
  archs=()
  [[ $destination == "visionOS Simulator" ]] && archs=(ARCHS=arm64)
  if ! xcodebuild build -scheme $NAME -destination "generic/platform=$destination" -configuration Release \
      -derivedDataPath $DD BUILD_LIBRARY_FOR_DISTRIBUTION=YES SKIP_INSTALL=NO $archs > $log 2>&1; then
    grep -E "error:" $log | sort -u | head -20
    print "failed: $log"
    exit 1
  fi
done

# Each slice's framework, with its module (interfaces) and resource bundle inside, where Bundle.module finds it.
ASSEMBLED=$DD/assembled
mkdir -p $ASSEMBLED
arguments=()
for slice in $SLICES; do
  suffix=${slice#*:}
  products=$DD/Build/Products/Release$suffix
  slice_dir=$ASSEMBLED/release$suffix
  if [[ -e $slice_dir ]]; then rm -rf "${slice_dir:?}"; fi
  mkdir -p $slice_dir
  cp -R $products/PackageFrameworks/$NAME.framework $slice_dir/
  framework=$slice_dir/$NAME.framework
  if [[ -d $framework/Versions ]]; then
    contents=$framework/Versions/A
    mkdir -p $contents/Modules $contents/Resources
    cp -R $products/$NAME.swiftmodule $contents/Modules/
    cp -R $products/${NAME}_$NAME.bundle $contents/Resources/
    ln -sfh Versions/Current/Modules $framework/Modules
  else
    mkdir -p $framework/Modules
    cp -R $products/$NAME.swiftmodule $framework/Modules/
    cp -R $products/${NAME}_$NAME.bundle $framework/
  fi
  arguments+=(-framework $framework -debug-symbols $products/$NAME.framework.dSYM)
done

# The tvOS device slice's archive: the upscaler's pipelines, from the very default.metallib the slice ships
# (an archive entry names its library's functions), recorded by a Metal 4 compiler on this Mac, then translated.
print "== archive (tvOS $TVOS_ARCHIVE_OS: ${ARCHIVE_GPUS[*]})"
swift build -c release --product Harvest -q
bundle=$ASSEMBLED/release-appletvos/$NAME.framework/${NAME}_$NAME.bundle
script=$DD/pipelines.mtl4-json
$DEV/.build/release/Harvest $bundle/default.metallib $script
gpus=()
for gpu in $ARCHIVE_GPUS; do gpus+=(-arch $gpu); done
xcrun metal-tt $gpus -tvos-version-min $TVOS_ARCHIVE_OS $bundle/default.metallib $script -o $bundle/$NAME-pipelines.metallib

mkdir -p $ROOT/Frameworks
output=$ROOT/Frameworks/$NAME.xcframework
if [[ -e $output ]]; then rm -rf "${output:?}"; fi
xcodebuild -create-xcframework $arguments -output $output
print "wrote $output"
