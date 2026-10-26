# Regenerating the models

Builds `Sources/LibMLSubUpscaler/Models/ArtCNN_<alpha>_<luma>.mlmodelc` (the compiled models the upscaler runs) and
`Tests/LibMLSubUpscalerTests/Weights/ArtCNN_<model>.f32` (the Float32 parameters the tests' reference reads) from
ArtCNN (https://github.com/Artoriuz/ArtCNN, commit 0837e5326e710f49e2ab0a8d995da0708047442e, MIT). Python 3 with
coremltools 8.3 and numpy, and Xcode for `coremlcompiler`. Run from the package root.

```sh
A=path/to/ArtCNN; W=work; M=Sources/LibMLSubUpscaler/Models
H=46,78,142; X=256,512,1024,2048   # window heights and widths (cores plus the 7-pixel reach on each side)
for m in C4F16 C4F32; do
  python3 Scripts/ArtCNN/parse_artcnn.py $A/GLSL/ArtCNN_$m.glsl $W/$m.json    # the GLSL hook's literals as layers
  python3 Scripts/ArtCNN/onnx_check.py $A/ONNX/ArtCNN_$m.onnx $W/$m.json      # they equal the ONNX initialisers
done
for a in C4F16 C4F32; do for l in C4F16 C4F32; do                             # a model per alpha / luma pairing
  python3 Scripts/ArtCNN/build_tiles.py $W/$a.json $W/$l.json $W/ArtCNN_${a}_$l.mlpackage $H $X
  xcrun coremlcompiler compile $W/ArtCNN_${a}_$l.mlpackage $M
done; done
python3 Scripts/ArtCNN/gen_weights.py Tests/LibMLSubUpscalerTests/Weights $W
```

The window sizes are `ArtCNNCoreMLModels.windowHeights` and `windowWidths`, which `SubtitleUpscaleTilePlan` covers a
picture with. The tests compare the compiled models' output with the Float32 reference, so a model that does not match
the weights fails them.
