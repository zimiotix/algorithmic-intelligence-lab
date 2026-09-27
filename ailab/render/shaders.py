"""GLSL 3.30 shaders. Everything is SDF-shaded, so shapes stay crisp at any zoom.

Hash functions: "Hash without Sine" (Dave Hoskins, MIT). SDF formulas: standard
distance functions as documented by Inigo Quilez (MIT).
"""

HEADER = """#version 330
uniform vec2 u_scale;
uniform vec2 u_offset;
uniform float u_px;      // world units per physical pixel
float cov(float d) { return 1.0 - smoothstep(-0.5 * u_px, 0.5 * u_px, d); }
"""

NOISE = """
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x),
               mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(17.1, 9.2); a *= 0.5; }
    return s;
}
vec2 worley(vec2 p, float t) {
    vec2 i = floor(p), f = fract(p);
    float f1 = 8.0, f2 = 8.0;
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        vec2 g = vec2(x, y);
        vec2 h = hash22(i + g);
        vec2 o = 0.5 + 0.42 * sin(t + 6.2831 * h);
        float d = length(g + o - f);
        if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
    }
    return vec2(f1, f2);
}
"""

SDF = """
float sdSeg(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}
float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
float sdEllipse(vec2 p, vec2 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / max(k1, 1e-6);
}
float sdTri(vec2 p, vec2 p0, vec2 p1, vec2 p2) {
    vec2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
    vec2 v0 = p - p0, v1 = p - p1, v2 = p - p2;
    vec2 q0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
    vec2 q1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
    vec2 q2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
    float s = sign(e0.x * e2.y - e0.y * e2.x);
    vec2 d = min(min(vec2(dot(q0, q0), s * (v0.x * e0.y - v0.y * e0.x)),
                     vec2(dot(q1, q1), s * (v1.x * e1.y - v1.y * e1.x))),
                     vec2(dot(q2, q2), s * (v2.x * e2.y - v2.y * e2.x)));
    return -sqrt(d.x) * sign(d.y);
}
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
vec2 rot(vec2 p, float a) { float c = cos(a), s = sin(a); return vec2(c*p.x - s*p.y, s*p.x + c*p.y); }
void over(inout vec4 acc, vec3 c, float a) {
    acc.rgb = c * a + acc.rgb * (1.0 - a);
    acc.a = a + acc.a * (1.0 - a);
}
"""

# ------------------------------------------------------------------ lines
LINES_VS = """
in vec2 in_corner;
in vec2 i_a; in vec2 i_b; in float i_w; in vec4 i_color;
out vec2 v_local; flat out float v_len; flat out float v_hw; flat out vec4 v_color;
void main() {
    vec2 d = i_b - i_a;
    float len = length(d);
    vec2 dir = len > 1e-6 ? d / len : vec2(1.0, 0.0);
    vec2 nrm = vec2(-dir.y, dir.x);
    float hw = max(i_w * 0.5, u_px * 0.5);
    float pad = hw + u_px * 1.5;
    float lx = mix(-pad, len + pad, in_corner.x * 0.5 + 0.5);
    float ly = in_corner.y * pad;
    v_local = vec2(lx, ly);
    v_len = len;
    v_hw = hw;
    v_color = vec4(i_color.rgb, i_color.a * clamp(i_w * 0.5 / hw, 0.0, 1.0));
    gl_Position = vec4((i_a + dir * lx + nrm * ly) * u_scale + u_offset, 0.0, 1.0);
}
"""
LINES_FS = """
in vec2 v_local; flat in float v_len; flat in float v_hw; flat in vec4 v_color;
out vec4 f_color;
void main() {
    float x = clamp(v_local.x, 0.0, v_len);
    float a = cov(length(v_local - vec2(x, 0.0)) - v_hw);
    if (a <= 0.002) discard;
    f_color = vec4(v_color.rgb, v_color.a * a);
}
"""

# ---------------------------------------------------------------- circles
CIRCLES_VS = """
in vec2 in_corner;
in vec2 i_c; in float i_r; in float i_ring; in float i_soft; in vec4 i_color;
out vec2 v_p; flat out float v_r; flat out float v_ring; flat out float v_soft; flat out vec4 v_color;
void main() {
    float ext = i_r + i_soft + u_px * 2.0;
    v_p = in_corner * ext;
    v_r = i_r; v_ring = i_ring; v_soft = i_soft; v_color = i_color;
    gl_Position = vec4((i_c + v_p) * u_scale + u_offset, 0.0, 1.0);
}
"""
CIRCLES_FS = """
in vec2 v_p; flat in float v_r; flat in float v_ring; flat in float v_soft; flat in vec4 v_color;
out vec4 f_color;
void main() {
    float d = length(v_p) - v_r;
    if (v_ring > 0.0) d = abs(d + v_ring * 0.5) - v_ring * 0.5;
    float a = cov(d);
    if (v_soft > 0.0) {
        float e = max(d, 0.0) / v_soft;
        a = max(v_r > 0.0 ? a : 0.0, exp(-4.0 * e * e));
    }
    if (a < 0.003) discard;
    f_color = vec4(v_color.rgb, v_color.a * a);
}
"""

# ---------------------------------------------------------------- sprites
SPRITES_VS = """
in vec2 in_corner;
in vec2 i_pos; in float i_angle; in vec2 i_size; in vec4 i_color; in float i_phase;
out vec2 v_q; flat out vec2 v_size; flat out vec4 v_color; flat out float v_phase;
void main() {
    float ext = max(i_size.x, i_size.y) * 0.62 + u_px * 2.0;
    v_q = in_corner * ext;
    v_size = i_size; v_color = i_color; v_phase = i_phase;
    gl_Position = vec4((i_pos + rot(v_q, i_angle)) * u_scale + u_offset, 0.0, 1.0);
}
"""
SPRITES_FS = """
uniform int u_kind;
in vec2 v_q; flat in vec2 v_size; flat in vec4 v_color; flat in float v_phase;
out vec4 f_color;

vec4 fish(vec2 q, float L, float W, vec3 base, float ph) {
    float t = clamp((0.15 * L - q.x) / (0.7 * L), 0.0, 1.0);
    q.y -= sin(ph - q.x / L * 3.0) * 0.32 * W * t * t;
    float body = sdEllipse(q - vec2(0.10 * L, 0.0), vec2(0.36 * L, 0.5 * W));
    float tail = sdTri(q, vec2(-0.16 * L, 0.0), vec2(-0.48 * L, 0.48 * W), vec2(-0.48 * L, -0.48 * W));
    float notch = length(q - vec2(-0.52 * L, 0.0)) - 0.16 * W;
    tail = max(tail, -notch);
    float d = smin(body, tail, 0.14 * W);
    vec4 acc = vec4(0.0);
    float a = cov(d);
    float flank = smoothstep(-0.30 * W, 0.0, d);
    vec3 c = base * (0.55 + 0.65 * flank);
    float ridge = exp(-pow(q.y / (0.14 * W), 2.0)) * smoothstep(-0.45 * L, 0.25 * L, q.x);
    c = mix(c, base * 0.30, 0.6 * ridge);
    c += vec3(0.12) * exp(-pow(length(q - vec2(0.30 * L, 0.0)) / (0.16 * L), 2.0));
    over(acc, c, a);
    float e = min(length(q - vec2(0.34 * L, 0.28 * W)), length(q - vec2(0.34 * L, -0.28 * W)))
              - (0.06 * W + 0.015 * L);
    over(acc, vec3(0.02, 0.03, 0.05), cov(e) * a);
    return acc;
}

vec4 predator(vec2 q, float L, float W, vec3 base, float ph) {
    float t = clamp((0.1 * L - q.x) / (0.7 * L), 0.0, 1.0);
    q.y -= sin(ph - q.x / L * 2.5) * 0.22 * W * t * t;
    float body = sdEllipse(q, vec2(0.40 * L, 0.46 * W));
    float nose = sdTri(q, vec2(0.52 * L, 0.0), vec2(0.25 * L, 0.34 * W), vec2(0.25 * L, -0.34 * W));
    body = smin(body, nose, 0.12 * W);
    vec2 fq = vec2(q.x, abs(q.y));
    float fin = sdTri(fq, vec2(0.14 * L, 0.30 * W), vec2(-0.04 * L, 0.30 * W), vec2(-0.14 * L, 1.15 * W));
    float tail = sdTri(q, vec2(-0.34 * L, 0.0), vec2(-0.60 * L, 0.62 * W), vec2(-0.56 * L, -0.50 * W));
    float d = smin(smin(body, fin, 0.10 * W), tail, 0.12 * W);
    vec4 acc = vec4(0.0);
    float a = cov(d);
    float spine = exp(-pow(q.y / (0.12 * W), 2.0));
    vec3 c = base * (0.75 + 0.45 * smoothstep(-0.35 * W, 0.0, d)) * (1.0 - 0.35 * spine);
    float rim = smoothstep(-0.10 * W, 0.0, d);
    c += vec3(1.6, 0.35, 0.18) * rim;
    over(acc, c, a);
    float e = min(length(q - vec2(0.30 * L, 0.22 * W)), length(q - vec2(0.30 * L, -0.22 * W))) - 0.05 * W;
    over(acc, vec3(2.2, 0.6, 0.3), cov(e) * a);
    return acc;
}

vec4 ant(vec2 q, float L, vec3 base, float ph) {
    float head = length(q - vec2(0.36 * L, 0.0)) - 0.10 * L;
    float thorax = sdEllipse(q - vec2(0.13 * L, 0.0), vec2(0.12 * L, 0.065 * L));
    float petiole = length(q - vec2(-0.02 * L, 0.0)) - 0.035 * L;
    float abdomen = sdEllipse(q - vec2(-0.25 * L, 0.0), vec2(0.20 * L, 0.13 * L));
    float body = min(min(head, thorax), min(petiole, abdomen));
    float legs = 1e9;
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        float x0 = (0.19 - 0.06 * fi) * L;
        for (int s = 0; s < 2; s++) {
            float side = s == 0 ? 1.0 : -1.0;
            float sw = sin(ph + 3.14159 * (mod(fi, 2.0) + float(s))) * 0.09 * L;
            vec2 base0 = vec2(x0, side * 0.04 * L);
            vec2 knee = vec2(x0 + (0.03 - 0.08 * fi) * L + sw * 0.5, side * 0.20 * L);
            vec2 tip = vec2(x0 + (0.08 - 0.17 * fi) * L + sw, side * 0.34 * L);
            legs = min(legs, min(sdSeg(q, base0, knee), sdSeg(q, knee, tip)) - 0.014 * L);
        }
    }
    vec2 aq = vec2(q.x, abs(q.y));
    float ant = min(sdSeg(aq, vec2(0.42 * L, 0.04 * L), vec2(0.52 * L, 0.11 * L)),
                    sdSeg(aq, vec2(0.52 * L, 0.11 * L), vec2(0.62 * L, 0.08 * L))) - 0.010 * L;
    vec4 acc = vec4(0.0);
    over(acc, base * 0.8, cov(min(legs, ant)));
    vec3 c = base * (0.8 + 0.5 * smoothstep(-0.06 * L, 0.0, body));
    c += vec3(0.30, 0.24, 0.18) * exp(-pow(length(q - vec2(-0.21 * L, 0.05 * L)) / (0.07 * L), 2.0));
    c += vec3(0.18, 0.15, 0.12) * exp(-pow(length(q - vec2(0.38 * L, 0.03 * L)) / (0.04 * L), 2.0));
    over(acc, c, cov(body));
    return acc;
}

vec4 car(vec2 q, float L, float W, vec3 base, float steer) {
    vec4 acc = vec4(0.0);
    vec2 wh = vec2(0.105 * L, 0.10 * W);
    float wheels = 1e9;
    for (int i = 0; i < 4; i++) {
        float fx = i < 2 ? 0.31 * L : -0.30 * L;
        float fy = (i % 2 == 0 ? 1.0 : -1.0) * 0.47 * W;
        vec2 p = q - vec2(fx, fy);
        if (i < 2) p = rot(p, -steer);
        wheels = min(wheels, sdBox(p, wh) - 0.02 * L);
    }
    over(acc, vec3(0.025), cov(wheels));
    vec2 bq = q;
    bq.y *= 1.0 + 0.45 * smoothstep(0.10 * L, 0.5 * L, q.x);
    float body = sdBox(bq, vec2(0.5 * L - 0.12 * W, 0.40 * W - 0.12 * W)) - 0.12 * W;
    vec3 c = base * (0.75 + 0.5 * smoothstep(-0.12 * W, 0.0, body));
    over(acc, c, cov(body));
    float stripe = max(abs(q.y) - 0.07 * W, body + 0.02 * W);
    over(acc, vec3(0.92, 0.94, 0.97), cov(stripe) * 0.9);
    float cock = sdEllipse(q - vec2(0.02 * L, 0.0), vec2(0.16 * L, 0.20 * W));
    vec3 glass = mix(vec3(0.02, 0.03, 0.05), vec3(0.25, 0.35, 0.45), smoothstep(0.1 * W, -0.15 * W, q.y));
    over(acc, glass, cov(cock));
    float wing = sdBox(q - vec2(-0.47 * L, 0.0), vec2(0.035 * L, 0.52 * W)) - 0.01 * L;
    over(acc, base * 0.35, cov(wing));
    float nosewing = sdBox(q - vec2(0.47 * L, 0.0), vec2(0.025 * L, 0.46 * W)) - 0.01 * L;
    over(acc, base * 0.5, cov(nosewing));
    return acc;
}

void main() {
    float L = v_size.x, W = v_size.y;
    vec3 base = v_color.rgb;
    vec2 q = v_q;
    vec4 acc = vec4(0.0);
    if (u_kind == 0) acc = fish(q, L, W, base, v_phase);
    else if (u_kind == 1) acc = predator(q, L, W, base, v_phase);
    else if (u_kind == 2) acc = ant(q, L, base, v_phase);
    else if (u_kind == 3) acc = car(q, L, W, base, v_phase);
    else if (u_kind == 4) {
        float d = sdTri(q, vec2(0.5 * L, 0.0), vec2(-0.5 * L, 0.5 * W), vec2(-0.5 * L, -0.5 * W));
        over(acc, base, cov(d));
    } else if (u_kind == 5) {
        float basep = sdBox(q, vec2(0.42 * L)) - 0.08 * L;
        over(acc, base * 0.55, cov(basep));
        float disc = length(q) - 0.33 * L;
        over(acc, base * (0.9 + 0.3 * smoothstep(0.0, -0.3 * L, disc)), cov(disc));
        float ring = abs(length(q) - 0.21 * L) - 0.045 * L;
        over(acc, vec3(1.3), cov(ring));
        over(acc, vec3(0.05), cov(length(q) - 0.07 * L));
    } else if (u_kind == 6) {
        float d = sdSeg(vec2(q.x, abs(q.y)), vec2(0.30 * L, 0.0), vec2(-0.30 * L, 0.45 * W)) - 0.10 * W;
        over(acc, base, cov(d));
    }
    if (acc.a < 0.003) discard;
    f_color = vec4(acc.rgb / max(acc.a, 1e-5), acc.a * v_color.a);
}
"""

# ------------------------------------------------------------------- mesh
MESH_VS = """
in vec2 in_pos; in vec4 in_color;
out vec2 v_world; out vec4 v_color;
void main() {
    v_world = in_pos; v_color = in_color;
    gl_Position = vec4(in_pos * u_scale + u_offset, 0.0, 1.0);
}
"""
MESH_FS = """
uniform int u_style;
in vec2 v_world; in vec4 v_color;
out vec4 f_color;
void main() {
    vec3 c = v_color.rgb;
    if (u_style == 1) {
        float n = vnoise(v_world * 0.35) * 0.45 + vnoise(v_world * 2.7) * 0.3
                + hash12(floor(v_world * 16.0)) * 0.25;
        c *= 0.80 + 0.32 * n;
    } else if (u_style == 2) {
        float n = fbm(v_world * 0.5);
        vec2 cr = worley(v_world * 0.9, 0.0);
        c *= (0.55 + 0.75 * n) * (0.8 + 0.3 * smoothstep(0.0, 0.15, cr.y - cr.x));
    }
    f_color = vec4(c, v_color.a);
}
"""

# ------------------------------------------------------------------ image
IMAGE_VS = """
in vec2 in_corner;
uniform vec4 u_bounds;
out vec2 v_uv;
out vec2 v_world;
void main() {
    v_uv = in_corner * 0.5 + 0.5;
    v_world = mix(u_bounds.xy, u_bounds.zw, v_uv);
    gl_Position = vec4(v_world * u_scale + u_offset, 0.0, 1.0);
}
"""
IMAGE_FS = """
uniform sampler2D u_tex;
uniform int u_mode;
in vec2 v_uv;
in vec2 v_world;
out vec4 f_color;
void main() {
    vec4 c = texture(u_tex, v_uv);
    if (u_mode == 1) {
        // Mask mode: the bilinear-filtered mask is a smooth field; threshold it for a
        // crisp organic outline, then shade: lit rim, noisy body, soft inner shadow.
        vec2 texel = 1.0 / vec2(textureSize(u_tex, 0));
        float m = c.a;
        float edge = fwidth(m) + 1e-4;
        float a = smoothstep(0.5 - edge, 0.5 + edge, m);
        if (a < 0.002) discard;
        float mx = texture(u_tex, v_uv + vec2(texel.x, 0)).a - texture(u_tex, v_uv - vec2(texel.x, 0)).a;
        float my = texture(u_tex, v_uv + vec2(0, texel.y)).a - texture(u_tex, v_uv - vec2(0, texel.y)).a;
        float light = clamp(0.5 - 0.9 * (mx * 0.6 + my * 0.8), 0.0, 1.0);
        float n = fbm(v_world * 0.7) * 0.6 + vnoise(v_world * 4.0) * 0.4;
        vec3 rock = c.rgb * (0.65 + 0.7 * n);
        rock *= mix(0.55, 1.35, light);
        rock *= mix(0.75, 1.0, smoothstep(0.5, 0.9, m));
        f_color = vec4(rock, a);
        return;
    }
    if (c.a < 0.002) discard;
    f_color = c;
}
"""

# ------------------------------------------------------------- fullscreen
FULLSCREEN_VS = """#version 330
out vec2 v_uv;
out vec2 v_ndc;
void main() {
    vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
    v_ndc = p;
    v_uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

BACKGROUND_FS = """
uniform int u_style;
uniform float u_time;
uniform vec4 u_world;
uniform vec3 u_tint;
in vec2 v_ndc;
out vec4 f_color;

vec3 water(vec2 w, float t, float gy) {
    vec3 deep = vec3(0.008, 0.035, 0.070), mid = vec3(0.020, 0.110, 0.155);
    float n = fbm(w * 0.02 + vec2(t * 0.01, 0.0));
    vec3 c = mix(deep, mid, clamp(0.30 + 0.55 * gy + 0.3 * (n - 0.5), 0.0, 1.0));
    vec2 warp = vec2(fbm(w * 0.025 + vec2(t * 0.020, 0.0)), fbm(w * 0.025 + vec2(5.2, -t * 0.020)));
    vec2 p = w * 0.055 + warp * 1.4;
    vec2 a = worley(p + vec2(0.0, t * 0.04), t * 0.5);
    vec2 b = worley(p * 1.8 + vec2(t * 0.03, 0.0), t * 0.7 + 1.7);
    float ca = pow(1.0 - smoothstep(0.0, 0.10, a.y - a.x), 3.0);
    float cb = pow(1.0 - smoothstep(0.0, 0.08, b.y - b.x), 3.0);
    float caust = (ca * 0.6 + cb * 0.4) * smoothstep(0.25, 0.75, fbm(w * 0.018 - t * 0.02));
    c += vec3(0.05, 0.15, 0.17) * caust * (0.35 + 0.65 * gy);
    float shaft = 0.5 + 0.5 * sin(w.x * 0.06 + w.y * 0.035 + t * 0.25) * sin(w.x * 0.023 - t * 0.11);
    c += vec3(0.020, 0.050, 0.060) * shaft * gy;
    // marine snow: sparse specks drifting slowly down
    vec2 sp = w * 0.9 + vec2(sin(t * 0.13) * 1.5, t * 0.35);
    vec2 cell = floor(sp);
    vec2 jit = hash22(cell);
    float speck = step(0.93, hash12(cell + 7.3)) * (1.0 - smoothstep(0.02, 0.06, length(fract(sp) - jit)));
    c += vec3(0.10, 0.16, 0.17) * speck;
    return c;
}
vec3 soil(vec2 w) {
    float n = fbm(w * 0.07);
    float n2 = fbm(w * 0.6 + 3.0);
    vec3 c = mix(vec3(0.060, 0.045, 0.032), vec3(0.120, 0.088, 0.058), n);
    c *= 0.82 + 0.32 * n2;
    vec2 pb = worley(w * 0.8, 0.0);
    float peb = (1.0 - smoothstep(0.08, 0.16, pb.x)) * step(0.62, hash12(floor(w * 0.8)));
    c += vec3(0.050, 0.040, 0.030) * peb;
    return c;
}
vec3 grass(vec2 w) {
    float stripe = smoothstep(0.45, 0.55, abs(fract(w.x / 12.0) - 0.5) * 2.0);
    float n = fbm(w * 0.12);
    vec3 c = mix(vec3(0.055, 0.125, 0.058), vec3(0.090, 0.185, 0.085), n);
    return c * (0.92 + 0.10 * stripe + 0.07 * vnoise(w * 5.0));
}
vec3 voidbg(vec2 w) {
    vec3 c = vec3(0.028, 0.038, 0.060);
    vec2 g = (fract(w / 5.0 + 0.5) - 0.5) * 5.0;
    float dotg = 1.0 - smoothstep(0.0, u_px * 1.5, length(g) - 0.10);
    return c + vec3(0.035, 0.050, 0.080) * dotg;
}
void main() {
    vec2 w = (v_ndc - u_offset) / u_scale;
    float gy = clamp((w.y - u_world.y) / (u_world.w - u_world.y), 0.0, 1.0);
    vec3 c;
    if (u_style == 1) c = water(w, u_time, gy);
    else if (u_style == 2) c = soil(w);
    else if (u_style == 3) c = grass(w);
    else c = voidbg(w);
    vec2 inside2 = smoothstep(u_world.xy - u_px * 2.0, u_world.xy, w)
                 * smoothstep(u_world.zw + u_px * 2.0, u_world.zw, w);
    float inside = inside2.x * inside2.y;
    c = mix(c * 0.35 + vec3(0.006, 0.008, 0.014), c, inside);
    f_color = vec4(c * u_tint, 1.0);
}
"""

# -------------------------------------------------------------------- post
BLOOM_DOWN_FS = """#version 330
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform int u_prefilter;
uniform float u_threshold;
in vec2 v_uv;
out vec4 f_color;
vec3 s(vec2 o) { return texture(u_src, v_uv + o * u_texel).rgb; }
void main() {
    vec3 a = s(vec2(-2, 2)), b = s(vec2(0, 2)), c = s(vec2(2, 2));
    vec3 d = s(vec2(-2, 0)), e = s(vec2(0, 0)), f = s(vec2(2, 0));
    vec3 g = s(vec2(-2, -2)), h = s(vec2(0, -2)), i = s(vec2(2, -2));
    vec3 j = s(vec2(-1, 1)), k = s(vec2(1, 1)), l = s(vec2(-1, -1)), m = s(vec2(1, -1));
    vec3 col = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625
             + (j + k + l + m) * 0.125;
    if (u_prefilter == 1) {
        float br = max(col.r, max(col.g, col.b));
        float knee = u_threshold * 0.5;
        float rq = clamp(br - u_threshold + knee, 0.0, 2.0 * knee);
        rq = rq * rq / (4.0 * knee + 1e-5);
        col *= max(rq, br - u_threshold) / max(br, 1e-5);
    }
    f_color = vec4(col, 1.0);
}
"""
BLOOM_UP_FS = """#version 330
uniform sampler2D u_src;
uniform vec2 u_texel;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 d = u_texel.xyxy * vec4(1.0, 1.0, -1.0, 0.0);
    vec3 s = texture(u_src, v_uv - d.xy).rgb;
    s += texture(u_src, v_uv - d.wy).rgb * 2.0;
    s += texture(u_src, v_uv - d.zy).rgb;
    s += texture(u_src, v_uv + d.zw).rgb * 2.0;
    s += texture(u_src, v_uv).rgb * 4.0;
    s += texture(u_src, v_uv + d.xw).rgb * 2.0;
    s += texture(u_src, v_uv + d.zy).rgb;
    s += texture(u_src, v_uv + d.wy).rgb * 2.0;
    s += texture(u_src, v_uv + d.xy).rgb;
    f_color = vec4(s / 16.0, 1.0);
}
"""
COMPOSITE_FS = """#version 330
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_k;
uniform vec2 u_res;
in vec2 v_uv;
out vec4 f_color;
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
void main() {
    vec3 c = texture(u_scene, v_uv).rgb + texture(u_bloom, v_uv).rgb * u_bloom_k;
    vec3 hi = 0.8 + 0.2 * (1.0 - exp(-(c - 0.8) / 0.2));
    c = mix(c, hi, step(0.8, c));
    vec2 q = v_uv - 0.5;
    q.x *= u_res.x / u_res.y;
    c *= mix(0.80, 1.0, smoothstep(1.15, 0.30, length(q)));
    c += (hash12(gl_FragCoord.xy) - 0.5) / 255.0;
    f_color = vec4(max(c, 0.0), 1.0);
}
"""
