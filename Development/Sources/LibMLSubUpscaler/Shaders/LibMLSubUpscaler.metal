// Metal source for `SubtitleUpscaler`'s GPU passes around the Neural Engine: the content mask, each window's model
// input (alpha and premultiplied luma), the composite of the models' outputs with bilinear chroma into a 2x
// premultiplied RGBA picture, the clear outside the kept area, and the Lanczos resample.
//
// Region. The passes run on the kept area: the content bounds (the whole picture without bounds) grown by S, the
// support radius (5 by default), within the picture. Output pixels farther than S from a pixel with non-zero alpha are
// exactly transparent: what ArtCNN adds there is its zero-input response (under 0.002, below an 8-bit step), the haze
// the cut removes. Coordinates below are the region's unless they say picture; texture reads add `origin` and read
// transparency outside the picture.

#include <metal_stdlib>
using namespace metal;

// Fast math: the passes around the Neural Engine are filtering and compositing, measured against the reference
// with a tolerance, not transcriptions to match bit for bit.
#pragma METAL fp math_mode(fast)

struct UpGeom {
    uint width;          // region width
    uint height;         // region height
    uint wordsX;         // ceil(width / 32)
    uint straightAlpha;  // the source's colour is not premultiplied
    int originX;         // picture position of the region
    int originY;
    uint pictureWidth;
    uint pictureHeight;
    uint support;        // S: output pixels farther than S from content are transparent (1...7)
};

// BT.709 luma weights. Any weights that sum to 1 work: chroma is colour minus luma, so it is recombined exactly.
constant float3 UP_LUMA = float3(0.2126f, 0.7152f, 0.0722f);

static inline float4 up_premultiplied(float4 c, constant UpGeom& g) {
    return g.straightAlpha != 0 ? float4(c.rgb * c.a, c.a) : c;
}

// The source at region position (x, y), premultiplied; transparent outside the picture.
static inline float4 up_source(texture2d<float, access::read> src, constant UpGeom& g, int x, int y) {
    const int px = x + g.originX, py = y + g.originY;
    if (px < 0 || py < 0 || px >= int(g.pictureWidth) || py >= int(g.pictureHeight)) return float4(0.0f);
    return up_premultiplied(src.read(uint2(uint(px), uint(py))), g);
}

// MARK: - Content mask

// Bit i of word (w, y): pixel (32w + i, y) has non-zero alpha. One thread per pixel; a simdgroup is one word (the
// threadgroup is 32 wide), gathered with a ballot.
kernel void up_mask_bits(texture2d<float, access::read> src [[texture(0)]],
                         device uint* bits [[buffer(0)]],
                         constant UpGeom& g [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    const bool row = gid.y < g.height;
    const bool set = row && gid.x < g.width && up_source(src, g, int(gid.x), int(gid.y)).a > 0.0f;
    const uint word = uint(uint64_t(simd_ballot(set)));
    if (lane == 0 && row && gid.x / 32 < g.wordsX) bits[gid.y * g.wordsX + gid.x / 32] = word;
}

// Row dilation by S: rows[y][w].
kernel void up_mask_rows(device const uint* bits [[buffer(0)]],
                         device uint* rows [[buffer(1)]],
                         constant UpGeom& g [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= g.wordsX || gid.y >= g.height) return;
    device const uint* row = bits + gid.y * g.wordsX;
    const uint m = row[gid.x];
    const uint l = gid.x > 0 ? row[gid.x - 1] : 0u;
    const uint r = gid.x + 1 < g.wordsX ? row[gid.x + 1] : 0u;
    const ulong left = (ulong(m) << 32) | ulong(l);
    const ulong right = (ulong(r) << 32) | ulong(m);
    uint h = m;
    for (uint k = 1; k <= g.support; k++) h |= uint(right >> k) | uint(left >> (32 - k));
    rows[gid.y * g.wordsX + gid.x] = h;
}

// Column dilation by S: support[y][w] has bit i set when pixel (32w + i, y) is within Chebyshev distance S of content,
// the composite's support.
kernel void up_mask_columns(device const uint* rows [[buffer(0)]],
                            device uint* support [[buffer(1)]],
                            constant UpGeom& g [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= g.wordsX || gid.y >= g.height) return;
    const int t = int(g.support);
    const int y0 = max(int(gid.y) - t, 0);
    const int y1 = min(int(gid.y) + t, int(g.height) - 1);
    uint acc = 0;
    for (int y = y0; y <= y1; y++) acc |= rows[uint(y) * g.wordsX + gid.x];
    support[gid.y * g.wordsX + gid.x] = acc;
}

// MARK: - Composite

// One source pixel's 2x2 output block: the networks' alpha and premultiplied luma (sub-pixel 2j + i), plus the
// source's chroma interpolated bilinearly (alpha-weighted, so the colour of transparent pixels never bleeds in) and
// scaled by the new alpha. Alpha is clamped to [0, 1] and colour to [0, alpha], which removes the networks' overshoot
// halos. (x, y) is a region position.
static inline void up_write_block(texture2d<float, access::read> src, texture2d<float, access::write> dst,
                                  constant UpGeom& g, uint x, uint y, half4 alpha, half4 luma) {
    const uint2 o = uint2(2 * (int(x) + g.originX), 2 * (int(y) + g.originY));
    float4 n[3][3];
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) n[dy + 1][dx + 1] = up_source(src, g, int(x) + dx, int(y) + dy);
    for (uint j = 0; j < 2; j++) {
        for (uint i = 0; i < 2; i++) {
            // Sub-pixel (i, j) sits a quarter pixel towards neighbour (2i - 1, 2j - 1): weights 3/4 and 1/4 per axis.
            const uint nx = 2 * i, ny = 2 * j;
            const float4 c = (9.0f * n[1][1] + 3.0f * n[1][nx] + 3.0f * n[ny][1] + n[ny][nx]) * (1.0f / 16.0f);
            const float3 chroma = c.a > (1.0f / 1024.0f) ? (c.rgb - dot(c.rgb, UP_LUMA)) / c.a : float3(0.0f);
            const float a = clamp(float(alpha[j * 2 + i]), 0.0f, 1.0f);
            // Both networks' outputs clamped to [0, 1], as ArtCNN's last step does.
            const float3 rgb = clamp(saturate(float(luma[j * 2 + i])) + a * chroma, 0.0f, a);
            dst.write(float4(rgb, a), o + uint2(i, j));
        }
    }
}

static inline bool up_supported(device const uint* support, constant UpGeom& g, uint x, uint y) {
    return ((support[y * g.wordsX + x / 32] >> (x % 32)) & 1u) != 0;
}

static inline void up_write_clear(texture2d<float, access::write> dst, constant UpGeom& g, uint x, uint y) {
    const uint2 o = uint2(2 * (int(x) + g.originX), 2 * (int(y) + g.originY));
    for (uint j = 0; j < 2; j++)
        for (uint i = 0; i < 2; i++) dst.write(float4(0.0f), o + uint2(i, j));
}

// MARK: - Neural Engine tiles

// One window of the kept area's grid (SubtitleUpscaleTilePlan): its picture position, and the part of its core the
// composite writes (the core within the kept area), in picture coordinates. The window is the core plus the
// network's 7-pixel reach, transparent where it leaves the picture.
struct UpWindow {
    int x;
    int y;
    int coreX;
    int coreY;
    uint coreWidth;
    uint coreHeight;
};

// One window's model input, alpha and premultiplied luma, into its two IOSurface textures. Dispatched over the window.
kernel void up_tile_gather(texture2d<float, access::read> src [[texture(0)]],
                           texture2d<half, access::write> alpha [[texture(1)]],
                           texture2d<half, access::write> luma [[texture(2)]],
                           constant UpGeom& g [[buffer(0)]],
                           constant UpWindow& w [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= alpha.get_width() || gid.y >= alpha.get_height()) return;
    const int2 p = int2(w.x, w.y) + int2(gid);
    float4 c = 0.0f;
    if (p.x >= 0 && p.y >= 0 && p.x < int(g.pictureWidth) && p.y < int(g.pictureHeight))
        c = up_premultiplied(src.read(uint2(p)), g);
    alpha.write(half4(half(c.a)), gid);
    luma.write(half4(half(dot(c.rgb, UP_LUMA))), gid);
}

// One window's composite over its core: the models' outputs are its two IOSurface textures, the four sub-pixel
// channels stacked, channel 2 * j + i (output pixel (2x + i, 2y + j)) from row (2 * j + i) * window height.
kernel void up_composite_tiles(texture2d<float, access::read> src [[texture(0)]],
                               texture2d<float, access::write> dst [[texture(1)]],
                               texture2d<half, access::read> alpha [[texture(2)]],
                               texture2d<half, access::read> luma [[texture(3)]],
                               device const uint* support [[buffer(0)]],
                               constant UpGeom& g [[buffer(1)]],
                               constant UpWindow& w [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= w.coreWidth || gid.y >= w.coreHeight) return;
    const int px = w.coreX + int(gid.x), py = w.coreY + int(gid.y);
    const uint x = uint(px - g.originX), y = uint(py - g.originY);
    if (!up_supported(support, g, x, y)) return up_write_clear(dst, g, x, y);
    const uint2 t = uint2(uint(px - w.x), uint(py - w.y));
    const uint h = alpha.get_height() / 4;
    half4 a, l;
    for (uint k = 0; k < 4; k++) {
        a[k] = alpha.read(t + uint2(0, k * h)).r;
        l[k] = luma.read(t + uint2(0, k * h)).r;
    }
    up_write_block(src, dst, g, x, y, a, l);
}

// Transparency for the 2x2 output blocks of a rectangle of source pixels (x, y, width, height): the pixels no
// window covers. One thread per source pixel.
kernel void up_clear_rect(texture2d<float, access::write> dst [[texture(0)]],
                          constant int4& r [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(r.z) || gid.y >= uint(r.w)) return;
    const uint2 o = 2 * (uint2(r.xy) + gid);
    for (uint j = 0; j < 2; j++)
        for (uint i = 0; i < 2; i++) dst.write(float4(0.0f), o + uint2(i, j));
}

// MARK: - Resample

static inline float up_lanczos3(float d) {
    d = fabs(d);
    if (d < 1e-5f) return 1.0f;
    if (d >= 3.0f) return 0.0f;
    const float p = M_PI_F * d;
    return 3.0f * sin(p) * sin(p / 3.0f) / (p * p);
}

// Separable Lanczos-3 on premultiplied colour, widened by the scale when reducing so it also low-passes. Outside the
// picture is transparent. `axis` 0 resamples x (dst width, src height), 1 resamples y; the y pass clamps.
kernel void up_resample(texture2d<float, access::read> src [[texture(0)]],
                        texture2d<float, access::write> dst [[texture(1)]],
                        constant uint& axis [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float srcLength = float(axis == 0 ? src.get_width() : src.get_height());
    const float dstLength = float(axis == 0 ? dst.get_width() : dst.get_height());
    const float scale = srcLength / dstLength;
    const float widen = max(scale, 1.0f);
    const float centre = (float(axis == 0 ? gid.x : gid.y) + 0.5f) * scale - 0.5f;
    const int lo = int(floor(centre - 3.0f * widen)) + 1;
    const int hi = int(floor(centre + 3.0f * widen));
    float4 sum = 0.0f;
    float weights = 0.0f;
    for (int k = lo; k <= hi; k++) {
        const float w = up_lanczos3((float(k) - centre) / widen);
        weights += w;
        if (k < 0 || k >= int(srcLength)) continue;
        sum += w * src.read(axis == 0 ? uint2(uint(k), gid.y) : uint2(gid.x, uint(k)));
    }
    float4 c = sum / weights;
    if (axis == 1) {
        c.a = clamp(c.a, 0.0f, 1.0f);
        c.rgb = clamp(c.rgb, 0.0f, c.a);
    }
    dst.write(c, gid);
}
