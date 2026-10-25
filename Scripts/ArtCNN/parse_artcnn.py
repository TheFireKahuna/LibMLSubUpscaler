# Parses an ArtCNN C4Fxx mpv GLSL hook file into layers:
# [{name, cin, cout, relu, bias[cout], w[cout][cin][ky][kx]}] (ky, kx = 0..2, tap offset = k - 1).
import re, sys, json, struct
src = open(sys.argv[1]).read()
blocks = src.split('//!DESC ')[1:]
layers = []
for b in blocks:
    name = b.split('\n', 1)[0].strip()
    if 'Depth-To-Space' in name:
        continue
    inits = re.findall(r'V4 result(\d+) = V4\(([^)]*)\);', b)
    nres = len(inits)
    cout = 4 * nres
    bias = [0.0] * cout
    for k, vals in inits:
        v = [float(x) for x in vals.split(',')]
        assert len(v) == 4
        for r in range(4):
            bias[4 * int(k) + r] = v[r]
    m = re.search(r'shared (F|V4) inp\[(\d+)\]', b)
    ngroups = int(m.group(2))
    scalar_in = m.group(1) == 'F'
    cin = 1 if scalar_in else 4 * ngroups
    w = [[[[None] * 3 for _ in range(3)] for _ in range(cin)] for _ in range(cout)]
    if scalar_in:
        terms = re.findall(r'result(\d+) \+= V4\(([^)]*)\) \* inp_(\d+)_(\d)_(\d);', b)
        for k, vals, g, x, y in terms:
            v = [float(t) for t in vals.split(',')]
            for r in range(4):
                assert w[4 * int(k) + r][0][int(y)][int(x)] is None
                w[4 * int(k) + r][0][int(y)][int(x)] = v[r]
    else:
        terms = re.findall(r'result(\d+) \+= M4\(([^)]*)\) \* inp_(\d+)_(\d)_(\d);', b)
        for k, vals, g, x, y in terms:
            v = [float(t) for t in vals.split(',')]
            assert len(v) == 16
            # GLSL mat4 is column-major: column j (v[4j..4j+3]) multiplies input component j.
            for j in range(4):
                for r in range(4):
                    o = 4 * int(k) + r; i = 4 * int(g) + j
                    assert w[o][i][int(y)][int(x)] is None
                    w[o][i][int(y)][int(x)] = v[4 * j + r]
    for o in range(cout):
        for i in range(cin):
            for y in range(3):
                for x in range(3):
                    assert w[o][i][y][x] is not None, (name, o, i, y, x)
    relu = 'max(result0, V4(0.0))' in b
    resid = b.count('_mul * texelFetch(conv2d_raw') > 0 and 'conv2d_5_mul' in b
    layers.append(dict(name=name, cin=cin, cout=cout, relu=relu, residual_input=resid, bias=bias, w=w))
for L in layers:
    print(L['name'], L['cin'], L['cout'], 'relu' if L['relu'] else '', 'resid-in' if L['residual_input'] else '', file=sys.stderr)
json.dump(layers, open(sys.argv[2], 'w'))
