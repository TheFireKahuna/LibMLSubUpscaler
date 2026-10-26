# LibMLSubUpscaler

Bitmap subtitles (PGS, DVB, DVD) upscaled 2x by ArtCNN on the Neural Engine, for Apple platforms. ArtCNN
(https://github.com/Artoriuz/ArtCNN, MIT, `ArtCNNLicense`) is a small convolutional network trained for line art; a
subtitle object's premultiplied RGBA crop goes in, and its upscale comes out on the GPU in a few milliseconds, the
video's GPU time left alone.

```swift
import LibMLSubUpscaler

SubtitleUpscaler.prewarm(.balanced)                     // at app open, in the background
let upscaler = try SubtitleUpscaler(configuration: .balanced, submission: gpu)  // the player's GPU submission
try upscaler.prepare(canvas: CGSize(width: 1920, height: 1080))
let sharp = try upscaler.upscale(image)                 // CGImage, CVPixelBuffer, or MTLTexture jobs
```

## What it does

ArtCNN is single channel, so the picture is split into its alpha and its premultiplied luma, each runs through a
network, and the colour comes back as the source's chroma, interpolated bilinearly and scaled by the new alpha, clamped
to [0, 1] and [0, alpha]. Output farther than 5 source pixels (S, the support) from content is exactly transparent.
Other sizes take one or two 2x passes and a Lanczos-3 resample. Configurations: C4F16, C4F32, balanced (C4F16 on
alpha, C4F32 on luma).

The network runs on the Neural Engine through Core ML. GPU passes prepare its input and composite its output: the
content mask (one plane, the support, over the kept area), each window's gather, each window's composite, and clears
of everything no window covers.

The upscaler owns no command queue. Its GPU passes go through the caller's submission (`SubtitleUpscaleSubmission`,
with `SubtitleUpscaleBatch` and `SubtitleUpscalePass`: pipelines, bindings by index, small constant arguments,
dispatches, barriers, and an event waited on before a batch or signalled after it), so they run on the player's queue
with its Metal, Metal 4 or Metal 3. The tests submit through a Metal 3 implementation (`Metal3Submission`).

## How a picture is run

1. **Plan** (`SubtitleUpscaleTilePlan`, CPU, 0.1 to 0.7 ms on the A15). The caller's `Content` gives each row's first
   and last opaque column (the `CGImage` and pixel buffer conveniences scan it). The rows are cut into bands, each one
   core height, covered by cores across only the columns their own support spans; rows with no support take no band.
   A dynamic programme over the rows picks the bands, each band's columns are the cheapest cover of its span, and
   every window is costed by its measured pair-call time.
2. **Windows.** The models have fixed shapes: heights 46, 78, 142 by widths 256, 512, 1024, 2048, each a core plus
   the network's 7-pixel reach on every side, so a core's output is exact.
3. **Every batch committed up front, no CPU waits between stages.** The clears and the mask, each window's gather in a
   batch that signals its window's event when it completes, and each window's composite in a batch that waits on its
   other event are all committed before any prediction runs.
4. **Predictions in flight.** One thread per window waits on its gather event, runs one Core ML call for both planes
   (a pair program: `xa`, `xl` in, `ya`, `yl` out), and signals its composite's event. Every event is signalled
   whatever fails.
5. **Zero copy.** Each window's four planes are IOSurface-backed `OneComponent16Half` pixel buffers, wrapped once as
   `r16Float` textures and once as `MLMultiArray(pixelBuffer:)`; the inputs and output backings are made once per
   surface set. A prediction that does not write its backing is an error.

**Loading.** Each function loads first with Core ML's default specialisation, which the system compiles for the
Neural Engine in about 160 ms a function when nothing is cached; `prewarm(_:)` and `prepare(canvas:)` return (or
the upscaler is usable) once those are in. Each is then replaced, a function at a time on a thread of its own, by its
`fastPrediction` specialisation, which takes 0.3 to 4 s a function cold. The system compiles one model at a time,
so loading functions side by side gains nothing (Mac, 12 threads against one: 18.1 s against 18.0 s
`fastPrediction`, 1.6 s against 1.8 s default). On the Mac the two specialisations predict within 10% of each
other; on the A15 not measured yet. `SubtitleUpscaleNeuralEngine.coldLoadReport()` measures both from a model path
the system has not seen.

Models: one compiled multifunction model per alpha / luma pairing (`Sources/LibMLSubUpscaler/Models`), a function
per window shape, weights stored once, built by `Scripts/ArtCNN` (its README has the steps). They load from the
package's resources with the `fastPrediction` specialisation; nothing is compiled or written at run time but what the
system does for the Neural Engine.

## Costs (Apple TV 4K, A15, tvOS 27.2, Release)

One-time, at app open (`prewarm(_:)`, then `prepare(canvas:)` on the upscaler that will run):

| | |
|---|---|
| GPU pipelines | 5 to 10 ms (about 1 s when the system shader cache is cold) |
| Neural Engine, one pairing, cold (install, OS update, purged cache), `fastPrediction` | about 12 s, nearly all function loads |
| the same, warm | about 0.5 s |
| `prepare(canvas:)` | 25 to 90 ms |

Per picture, steady state, p50 (p99 within about 1 ms):

| Picture | C4F16 | balanced | C4F32 |
|---|---|---|---|
| 900x80, one line | 3.7 ms | 4.3 ms | 5.0 ms |
| 1400x120 strip | 3.7 | 6.8 | 8.3 |
| 1500x150, two lines of different lengths | 6.9 | 7.6 | 8.0 |
| 1920x200 strip | 7.6 | 10.6 | 12.2 |

Of a call, about 70 to 75% is the Neural Engine. The first picture on a prepared upscaler is still 1 to 4 ms above
steady state on some pictures. C4F16 alone is the least steady: its predictions vary run to run more than the
others'.

`SubtitleUpscaler.benchmark()` prints all of this for the device it runs on, the per-shape prediction times the plan's
cost table comes from included. Measure in Release only; Debug figures are not comparable (the planner alone runs 50
times slower). On a Mac:

```sh
SUBTITLE_UPSCALE_BENCH=1 swift test -c release -Xswiftc -enable-testing --filter SubtitleUpscaleBenchmarkTests
```

On a device, call `SubtitleUpscaler.benchmark()` from a test or probe built in the Release configuration.

## Findings

- **The Neural Engine, not the GPU.** An earlier GPU network was slower on the A15 in every configuration but C4F16
  on small pictures (even there), its p99 reached 1.5 times its median where the Neural Engine's stays within about a
  millisecond, and it competed with the video.
- **Core ML shapes.** A flexible shape is fast only at its default size; with infrequent reshapes each change
  re-optimises for about 10 ms. Fixed shapes in a multifunction model switch for free.
- **Per-pixel cost is near the Neural Engine's ceiling** for C4F32 (about 6 TFLOPS of fp16); C4F16's 16 channels use
  it less well. Repacking pixels into channels would add 4x the work as zero weights; the A15 gains nothing from int8.
- **No hand-over cost.** The same burst of predictions times the same after a GPU write into its surfaces, back to
  back, and on Core ML's own arrays; zero copy is kept on principle and costs nothing.
- **The CPU's clock matters, QoS does not.** The predict stage was a third shorter with another core kept busy;
  utility, default and user-interactive QoS made no difference. Removing the CPU's waits (steps 3 and 4 above) took
  most of that.
- **A richer shape library is worth 3 to 5%** (80 functions against 12, modelled over 100 subtitle objects), less
  than its load time.
- **Core AI (tvOS 27) cannot help on the A15**: `ComputeUnitKind.availableKinds` is `[cpu]` there; preferring the
  Neural Engine fails to specialise, and the CPU runs ArtCNN at about 30 ns a pixel. Ahead-of-time compilation is
  documented only for devices with Apple Intelligence. Revisit on newer Apple TV hardware.
- **Metal 4's machine-learning pass** builds an MPSGraph package (`metal-package-builder`), most likely GPU work;
  not measured.

## Open

- The first picture after `prepare(canvas:)`: the remaining 1 to 4 ms.
- Single-window pictures cannot overlap their stages; a split worth its extra call could be chosen by the planner.
- The plan's CPU time (up to 0.7 ms) is on the critical path.
- Under playback load (4K HDR decode, a player's GPU work) nothing is measured yet.

## Tests

`swift test` checks the upscale against a Float32 reference of ArtCNN (`Tests/LibMLSubUpscalerTests/ArtCNNNetwork.swift`,
reading `Weights/`), exact transparency away from content, the planner's cover, and the conveniences.

## Licence

MIT (`LICENSE`). The models and the reference weights are derived from ArtCNN, MIT; its notice is
`ArtCNNLicense.text`, for a host that shows third-party notices.
