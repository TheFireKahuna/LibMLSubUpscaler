import sys, struct, json
def varint(b, i):
    r = 0; s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7f) << s; s += 7
        if c < 0x80: return r, i
def fields(b):
    i = 0; out = []
    while i < len(b):
        k, i = varint(b, i); f, t = k >> 3, k & 7
        if t == 0: v, i = varint(b, i)
        elif t == 1: v = b[i:i+8]; i += 8
        elif t == 5: v = b[i:i+4]; i += 4
        elif t == 2:
            n, i = varint(b, i); v = b[i:i+n]; i += n
        else: raise Exception(t)
        out.append((f, t, v))
    return out
model = fields(open(sys.argv[1], 'rb').read())
graph = [v for f, t, v in model if f == 7][0]
g = fields(graph)
inits = {}
for f, t, v in g:
    if f == 5:
        tf = fields(v); dims = []; name = ''; raw = b''; fl = []
        for ff, tt, vv in tf:
            if ff == 1:
                if tt == 0: dims.append(vv)
                else:
                    j = 0
                    while j < len(vv): d, j = varint(vv, j); dims.append(d)
            elif ff == 8: name = vv.decode()
            elif ff == 9: raw = vv
            elif ff == 4:
                fl += list(struct.unpack('<%df' % (len(vv)//4), vv)) if tt == 2 else [struct.unpack('<f', vv)[0]]
        vals = list(struct.unpack('<%df' % (len(raw)//4), raw)) if raw else fl
        inits[name] = (dims, vals)
    if f == 1:
        nf = fields(v)
        op = [vv.decode() for ff, tt, vv in nf if ff == 4]
        ins = [vv.decode() for ff, tt, vv in nf if ff == 1]
        print('node', op, ins, file=sys.stderr)
for n, (d, v) in inits.items(): print('init', n, d, len(v), file=sys.stderr)
layers = json.load(open(sys.argv[2]))
# match conv weights by shape order: ONNX Conv W = [Cout, Cin, kH, kW]
convw = [(n, d, v) for n, (d, v) in inits.items() if len(d) == 4]
convb = [(n, d, v) for n, (d, v) in inits.items() if len(d) == 1]
maxd = 0
for L, (n, d, v), (nb, db, vb) in zip(layers, convw, convb):
    co, ci = L['cout'], L['cin']
    if d[0] != co:  # maybe HWIO
        print('shape', n, d)
    for o in range(co):
        for i in range(ci):
            for y in range(3):
                for x in range(3):
                    if d[0] == co: ref = v[((o*ci + i)*3 + y)*3 + x]
                    else: ref = v[((y*3 + x)*ci + i)*co + o]
                    maxd = max(maxd, abs(ref - L['w'][o][i][y][x]))
        maxd = max(maxd, abs(vb[o] - L['bias'][o]))
print('max |onnx - glsl| =', maxd)
