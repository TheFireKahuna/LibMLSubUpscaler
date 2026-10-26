# ArtCNN as one fp16 multifunction MLProgram for the Neural Engine, both of a subtitle's planes in one call: a
# function per window shape "h<H>w<W>" with inputs xa (alpha) and xl (premultiplied luma), each (1, 1, H, W), run
# through the alpha and the luma network, and outputs ya and yl, each (1, 4, H, W): the last convolution's four
# sub-pixel channels (channel 2 * row + column of the 2x block), unclamped. The composite does the depth-to-space
# and the clamp. The functions share one weight blob.
# Usage: build_tiles.py alpha_layers.json luma_layers.json out.mlpackage heights widths   (comma-separated sizes)
import json, os, sys, tempfile, numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
nets = [json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))]
out = sys.argv[3]
heights = [int(v) for v in sys.argv[4].split(',')]; widths = [int(v) for v in sys.argv[5].split(',')]
params = [([np.array(L['w'], dtype=np.float16) for L in n], [np.array(L['bias'], dtype=np.float16) for L in n]) for n in nets]
def net(x, k, name):
    Ws, Bs = params[k]
    def conv(t, i): return mb.conv(x=t, weight=Ws[i], bias=Bs[i], pad_type="same")
    f0 = conv(x, 0); t = f0
    for i in range(1, 5): t = mb.relu(x=conv(t, i))
    t = mb.add(x=conv(t, 5), y=f0)
    return mb.identity(x=conv(t, 6), name=name)
def program(h, w):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, h, w), dtype=types.fp16), mb.TensorSpec(shape=(1, 1, h, w), dtype=types.fp16)],
                opset_version=ct.target.iOS18)
    def prog(xa, xl):
        return net(xa, 0, "ya"), net(xl, 1, "yl")
    return prog
work = tempfile.mkdtemp()
desc = ct.utils.MultiFunctionDescriptor()
for h in heights:
    for w in widths:
        path = os.path.join(work, 'h%dw%d.mlpackage' % (h, w))
        ct.convert(program(h, w), convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                   compute_precision=ct.precision.FLOAT16).save(path)
        desc.add_function(path, src_function_name="main", target_function_name='h%dw%d' % (h, w))
desc.default_function_name = 'h%dw%d' % (heights[0], widths[0])
ct.utils.save_multifunction(desc, out)
print("saved", out)
