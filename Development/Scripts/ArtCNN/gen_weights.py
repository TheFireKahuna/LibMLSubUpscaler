# Writes ArtCNN's parameters (the layers parse_artcnn.py read, <root>/<model>.json) as the test reference's
# Weights/ArtCNN_<model>.f32: little-endian Float32, per layer in network order the kernel as
# [output][input][row][column] (ONNX OIHW) followed by the bias. Usage: gen_weights.py outdir root
import json, struct, sys, hashlib
out, root = sys.argv[1], sys.argv[2]
for m in ('C4F16', 'C4F32'):
    buf = b''
    for L in json.load(open(root + '/' + m + '.json')):
        for o in range(L['cout']):
            for i in range(L['cin']):
                for y in range(3):
                    for x in range(3):
                        buf += struct.pack('<f', L['w'][o][i][y][x])
        buf += b''.join(struct.pack('<f', L['bias'][o]) for o in range(L['cout']))
    open('%s/ArtCNN_%s.f32' % (out, m), 'wb').write(buf)
    print(m, len(buf) // 4, 'floats, SHA-256', hashlib.sha256(buf).hexdigest()[:16])
