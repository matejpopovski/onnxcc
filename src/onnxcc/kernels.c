/* Kernel library for onnxcc. Sections between @kernel and @end markers are copied into generated
 * files on demand, so a model pays only for the kernels it uses. A `generic` section is written
 * for float and instantiated for int8 by textual substitution. This file is also a valid C99
 * translation unit, which is how the test suite keeps every kernel compiling and warning-free.
 *
 * Rules every kernel follows:
 *   - no libc, no libm, no recursion, no variable-length arrays
 *   - the output pointer is the first argument
 *   - array indexing goes through int32_t offsets, so no out-of-range pointer is ever formed
 *   - `restrict` appears only where the planner is forbidden from sharing storage
 */

/*@kernel prelude*/
#include <stdint.h>

#define OX_MAX_RANK 8
/*@end*/

/*@kernel copy_f32 generic*/
static void ox_copy_f32(float *restrict y, const float *restrict x, int32_t n)
{
    for (int32_t i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}
/*@end*/

/*@kernel walk_t*/
/* Row-major loop nest over `dims`, where `strides[d]` is the signed source step of axis `d`. */
typedef struct {
    int32_t rank;
    int32_t start;
    const int32_t *dims;
    const int32_t *strides;
} ox_walk_t;

/* A loop on purpose: `= {0}` on an array lets the compiler emit a call to memset. */
static void ox_zero(int32_t *idx, int32_t rank)
{
    for (int32_t d = 0; d < rank; ++d) {
        idx[d] = 0;
    }
}

/* Advance the odometer. Returns 0 once every outer index has wrapped around. */
static int32_t ox_next(int32_t *idx, int32_t *offset, const ox_walk_t *p)
{
    for (int32_t d = p->rank - 2; d >= 0; --d) {
        *offset += p->strides[d];
        if (++idx[d] < p->dims[d]) {
            return 1;
        }
        *offset -= p->strides[d] * p->dims[d];
        idx[d] = 0;
    }
    return 0;
}
/*@end*/

/*@kernel gather_f32 generic requires=walk_t*/
/* One kernel serves Transpose and Slice: both read the source along a strided loop nest. */
static void ox_gather_f32(float *restrict y, const float *restrict x, const ox_walk_t *p)
{
    int32_t idx[OX_MAX_RANK];
    int32_t offset = p->start;
    const int32_t inner = p->dims[p->rank - 1];
    const int32_t step = p->strides[p->rank - 1];
    ox_zero(idx, p->rank);
    do {
        for (int32_t i = 0; i < inner; ++i) {
            y[i] = x[offset + i * step];
        }
        y += inner;
    } while (ox_next(idx, &offset, p));
}
/*@end*/

/*@kernel pad_f32 generic requires=walk_t*/
/* `p` walks the output region that receives the input. Everything else holds `value`. */
static void ox_pad_f32(float *restrict y, const float *restrict x, float value, int32_t size,
                       const ox_walk_t *p)
{
    int32_t idx[OX_MAX_RANK];
    int32_t offset = p->start;
    const int32_t inner = p->dims[p->rank - 1];
    ox_zero(idx, p->rank);
    for (int32_t i = 0; i < size; ++i) {
        y[i] = value;
    }
    do {
        for (int32_t i = 0; i < inner; ++i) {
            y[offset + i] = x[i];
        }
        x += inner;
    } while (ox_next(idx, &offset, p));
}
/*@end*/

/*@kernel concat_f32 generic*/
/* Each input contributes `chunks[j]` elements per outer block, in input order. */
static void ox_concat_f32(float *restrict y, const float *const *inputs, const int32_t *chunks,
                          int32_t count, int32_t blocks)
{
    for (int32_t block = 0; block < blocks; ++block) {
        for (int32_t j = 0; j < count; ++j) {
            const float *x = inputs[j] + block * chunks[j];
            for (int32_t i = 0; i < chunks[j]; ++i) {
                y[i] = x[i];
            }
            y += chunks[j];
        }
    }
}
/*@end*/

/*@kernel window_t*/
typedef struct {
    int32_t n, c_in, h_in, w_in, c_out, h_out, w_out;
    int32_t k_h, k_w, s_h, s_w, d_h, d_w, p_t, p_l, group;
} ox_window_t;
/*@end*/

/*@kernel taps*/
/* Limit window taps first + t * dilation, for t in [0, k), to the valid range [0, limit). */
static void ox_taps(int32_t first, int32_t k, int32_t dilation, int32_t limit, int32_t *lo,
                    int32_t *hi)
{
    const int32_t last = first + (k - 1) * dilation;
    *lo = first >= 0 ? 0 : (dilation - 1 - first) / dilation;
    if (last < limit) {
        *hi = k;
    } else if (first >= limit) {
        *hi = 0;
    } else {
        *hi = (limit - 1 - first) / dilation + 1;
    }
}
/*@end*/

/*@kernel span*/
/* Output indices [lo, hi) for which first + index * stride stays inside [0, limit). */
static void ox_span(int32_t first, int32_t stride, int32_t limit, int32_t count, int32_t *lo,
                    int32_t *hi)
{
    const int32_t room = limit - 1 - first;
    *lo = first >= 0 ? 0 : (stride - 1 - first) / stride;
    *hi = room < 0 ? 0 : room / stride + 1;
    *hi = *hi < count ? *hi : count;
}
/*@end*/

/*@kernel conv2d_f32 requires=window_t,span*/
/* Scatter form: each weight adds a shifted copy of an input row to the output row. Inner loops
 * run over whole rows, which is the shape auto-vectorizers handle well. */
static void ox_conv2d_f32(float *restrict y, const float *restrict x, const float *restrict w,
                          const float *restrict b, const ox_window_t *p, float lo, float hi)
{
    const int32_t plane_out = p->h_out * p->w_out;
    const int32_t plane_in = p->h_in * p->w_in;
    const int32_t cg_in = p->c_in / p->group;
    const int32_t cg_out = p->c_out / p->group;
    for (int32_t n = 0; n < p->n; ++n) {
        const float *x_batch = x + n * p->c_in * plane_in;
        float *y_batch = y + n * p->c_out * plane_out;
        for (int32_t c = 0; c < p->c_out; ++c) {
            const float *wc = w + c * cg_in * p->k_h * p->k_w;
            const float *xg = x_batch + (c / cg_out) * cg_in * plane_in;
            float *yp = y_batch + c * plane_out;
            const float bias = b ? b[c] : 0.0f;
            for (int32_t i = 0; i < plane_out; ++i) {
                yp[i] = bias;
            }
            for (int32_t ic = 0; ic < cg_in; ++ic) {
                const float *xp = xg + ic * plane_in;
                for (int32_t kh = 0; kh < p->k_h; ++kh) {
                    const int32_t off_h = kh * p->d_h - p->p_t;
                    int32_t oy_lo, oy_hi;
                    ox_span(off_h, p->s_h, p->h_in, p->h_out, &oy_lo, &oy_hi);
                    for (int32_t kw = 0; kw < p->k_w; ++kw) {
                        const float weight = *wc++;
                        const int32_t off_w = kw * p->d_w - p->p_l;
                        int32_t ox_lo, ox_hi;
                        ox_span(off_w, p->s_w, p->w_in, p->w_out, &ox_lo, &ox_hi);
                        const int32_t count = ox_hi - ox_lo;
                        if (count <= 0) {
                            continue;
                        }
                        for (int32_t oy = oy_lo; oy < oy_hi; ++oy) {
                            const int32_t iy = oy * p->s_h + off_h;
                            const float *row = xp + iy * p->w_in + ox_lo * p->s_w + off_w;
                            float *out = yp + oy * p->w_out + ox_lo;
                            if (p->s_w == 1) {
                                for (int32_t i = 0; i < count; ++i) {
                                    out[i] += weight * row[i];
                                }
                            } else {
                                for (int32_t i = 0; i < count; ++i) {
                                    out[i] += weight * row[i * p->s_w];
                                }
                            }
                        }
                    }
                }
            }
            if (lo <= hi) {
                for (int32_t i = 0; i < plane_out; ++i) {
                    yp[i] = yp[i] < lo ? lo : (yp[i] > hi ? hi : yp[i]);
                }
            }
        }
    }
}
/*@end*/

/*@kernel dense_f32*/
/* y[m][n] = a[m][k] @ w[k][n] + b. The ikj loop order keeps the inner loop vectorizable. */
static void ox_dense_f32(float *restrict y, const float *restrict a, const float *restrict w,
                         const float *restrict b, int32_t m, int32_t k, int32_t n, float lo,
                         float hi)
{
    for (int32_t r = 0; r < m; ++r) {
        float *row = y + r * n;
        for (int32_t j = 0; j < n; ++j) {
            row[j] = b ? b[j] : 0.0f;
        }
        for (int32_t i = 0; i < k; ++i) {
            const float value = a[r * k + i];
            const float *wi = w + i * n;
            for (int32_t j = 0; j < n; ++j) {
                row[j] += value * wi[j];
            }
        }
        if (lo <= hi) {
            for (int32_t j = 0; j < n; ++j) {
                row[j] = row[j] < lo ? lo : (row[j] > hi ? hi : row[j]);
            }
        }
    }
}
/*@end*/

/*@kernel maxpool_f32 generic requires=window_t,taps*/
static void ox_maxpool_f32(float *restrict y, const float *restrict x, const ox_window_t *p)
{
    const int32_t planes = p->n * p->c_in;
    for (int32_t c = 0; c < planes; ++c) {
        const float *xp = x + c * p->h_in * p->w_in;
        for (int32_t oy = 0; oy < p->h_out; ++oy) {
            const int32_t top = oy * p->s_h - p->p_t;
            int32_t kh_lo, kh_hi;
            ox_taps(top, p->k_h, p->d_h, p->h_in, &kh_lo, &kh_hi);
            for (int32_t ox = 0; ox < p->w_out; ++ox) {
                const int32_t left = ox * p->s_w - p->p_l;
                int32_t kw_lo, kw_hi;
                ox_taps(left, p->k_w, p->d_w, p->w_in, &kw_lo, &kw_hi);
                float best = xp[(top + kh_lo * p->d_h) * p->w_in + left + kw_lo * p->d_w];
                for (int32_t kh = kh_lo; kh < kh_hi; ++kh) {
                    const int32_t row = (top + kh * p->d_h) * p->w_in + left;
                    for (int32_t kw = kw_lo; kw < kw_hi; ++kw) {
                        const float value = xp[row + kw * p->d_w];
                        best = value > best ? value : best;
                    }
                }
                *y++ = best;
            }
        }
    }
}
/*@end*/

/*@kernel avgpool_f32 requires=window_t,taps*/
static void ox_avgpool_f32(float *restrict y, const float *restrict x, const ox_window_t *p,
                           int32_t include_pad)
{
    const int32_t planes = p->n * p->c_in;
    for (int32_t c = 0; c < planes; ++c) {
        const float *xp = x + c * p->h_in * p->w_in;
        for (int32_t oy = 0; oy < p->h_out; ++oy) {
            const int32_t top = oy * p->s_h - p->p_t;
            int32_t kh_lo, kh_hi;
            ox_taps(top, p->k_h, p->d_h, p->h_in, &kh_lo, &kh_hi);
            for (int32_t ox = 0; ox < p->w_out; ++ox) {
                const int32_t left = ox * p->s_w - p->p_l;
                int32_t kw_lo, kw_hi;
                ox_taps(left, p->k_w, p->d_w, p->w_in, &kw_lo, &kw_hi);
                float sum = 0.0f;
                for (int32_t kh = kh_lo; kh < kh_hi; ++kh) {
                    const int32_t row = (top + kh * p->d_h) * p->w_in + left;
                    for (int32_t kw = kw_lo; kw < kw_hi; ++kw) {
                        sum += xp[row + kw * p->d_w];
                    }
                }
                const int32_t count =
                    include_pad ? p->k_h * p->k_w : (kh_hi - kh_lo) * (kw_hi - kw_lo);
                *y++ = sum / (float)count;
            }
        }
    }
}
/*@end*/

/*@kernel gap_f32*/
static void ox_gap_f32(float *restrict y, const float *restrict x, int32_t planes, int32_t size)
{
    const float divisor = (float)size;
    for (int32_t c = 0; c < planes; ++c) {
        float sum = 0.0f;
        for (int32_t i = 0; i < size; ++i) {
            sum += x[c * size + i];
        }
        y[c] = sum / divisor;
    }
}
/*@end*/

/*@kernel reduce_f32 requires=walk_t*/
/* `p` walks the input in row-major order while its strides address the reduced output. */
static void ox_reduce_f32(float *restrict y, const float *restrict x, const ox_walk_t *p,
                          int32_t out_size, float divisor)
{
    int32_t idx[OX_MAX_RANK];
    int32_t offset = 0;
    const int32_t inner = p->dims[p->rank - 1];
    const int32_t step = p->strides[p->rank - 1];
    ox_zero(idx, p->rank);
    for (int32_t i = 0; i < out_size; ++i) {
        y[i] = 0.0f;
    }
    do {
        for (int32_t i = 0; i < inner; ++i) {
            y[offset + i * step] += x[i];
        }
        x += inner;
    } while (ox_next(idx, &offset, p));
    for (int32_t i = 0; i < out_size; ++i) {
        y[i] /= divisor;
    }
}
/*@end*/

/*@kernel bcast_t*/
/* Broadcasting loop nest. Axes where both operands are contiguous are merged by the compiler,
 * so an element-wise operation on equal shapes arrives here with rank 1. */
typedef struct {
    int32_t rank;
    const int32_t *dims;
    const int32_t *sa;
    const int32_t *sb;
} ox_bcast_t;

#define OX_BINARY(NAME, TYPE, QTYPE, EXPR)                                                         \
    static void NAME(TYPE *y, const TYPE *a, const TYPE *b, const ox_bcast_t *p, const QTYPE *q)   \
    {                                                                                              \
        int32_t idx[OX_MAX_RANK];                                                                  \
        const int32_t inner = p->dims[p->rank - 1];                                                \
        const int32_t sa = p->sa[p->rank - 1];                                                     \
        const int32_t sb = p->sb[p->rank - 1];                                                     \
        int32_t ia = 0;                                                                            \
        int32_t ib = 0;                                                                            \
        int32_t d;                                                                                 \
        (void)q;                                                                                   \
        for (d = 0; d < p->rank; ++d) {                                                            \
            idx[d] = 0;                                                                            \
        }                                                                                          \
        do {                                                                                       \
            if (sa != 0 && sb != 0) {                                                              \
                for (int32_t i = 0; i < inner; ++i) {                                              \
                    const TYPE u = a[ia + i];                                                      \
                    const TYPE v = b[ib + i];                                                      \
                    y[i] = EXPR;                                                                   \
                }                                                                                  \
            } else {                                                                               \
                for (int32_t i = 0; i < inner; ++i) {                                              \
                    const TYPE u = a[ia + i * sa];                                                 \
                    const TYPE v = b[ib + i * sb];                                                 \
                    y[i] = EXPR;                                                                   \
                }                                                                                  \
            }                                                                                      \
            y += inner;                                                                            \
            for (d = p->rank - 2; d >= 0; --d) {                                                   \
                ia += p->sa[d];                                                                    \
                ib += p->sb[d];                                                                    \
                if (++idx[d] < p->dims[d]) {                                                       \
                    break;                                                                         \
                }                                                                                  \
                ia -= p->sa[d] * p->dims[d];                                                       \
                ib -= p->sb[d] * p->dims[d];                                                       \
                idx[d] = 0;                                                                        \
            }                                                                                      \
        } while (d >= 0);                                                                          \
    }
/*@end*/

/*@kernel add_f32 requires=bcast_t*/
OX_BINARY(ox_add_f32, float, void, u + v)
/*@end*/

/*@kernel sub_f32 requires=bcast_t*/
OX_BINARY(ox_sub_f32, float, void, u - v)
/*@end*/

/*@kernel mul_f32 requires=bcast_t*/
OX_BINARY(ox_mul_f32, float, void, u *v)
/*@end*/

/*@kernel div_f32 requires=bcast_t*/
OX_BINARY(ox_div_f32, float, void, u / v)
/*@end*/

/*@kernel min_f32 requires=bcast_t*/
OX_BINARY(ox_min_f32, float, void, u < v ? u : v)
/*@end*/

/*@kernel max_f32 requires=bcast_t*/
OX_BINARY(ox_max_f32, float, void, u > v ? u : v)
/*@end*/

/*@kernel prelu_f32 requires=bcast_t*/
OX_BINARY(ox_prelu_f32, float, void, u < 0.0f ? v * u : u)
/*@end*/

/*@kernel unary_t*/
#define OX_UNARY(NAME, EXPR)                                                                       \
    static void NAME(float *y, const float *x, int32_t n, float p0, float p1)                      \
    {                                                                                              \
        (void)p0;                                                                                  \
        (void)p1;                                                                                  \
        for (int32_t i = 0; i < n; ++i) {                                                          \
            const float v = x[i];                                                                  \
            y[i] = EXPR;                                                                           \
        }                                                                                          \
    }
/*@end*/

/*@kernel clip_f32 requires=unary_t*/
OX_UNARY(ox_clip_f32, v < p0 ? p0 : (v > p1 ? p1 : v))
/*@end*/

/*@kernel abs_f32 requires=unary_t*/
OX_UNARY(ox_abs_f32, v < 0.0f ? -v : v)
/*@end*/

/*@kernel neg_f32 requires=unary_t*/
OX_UNARY(ox_neg_f32, -v)
/*@end*/

/*@kernel leaky_relu_f32 requires=unary_t*/
OX_UNARY(ox_leaky_relu_f32, v < 0.0f ? p0 * v : v)
/*@end*/

/*@kernel hard_sigmoid_scalar*/
static float ox_hard_sigmoid(float v, float alpha, float beta)
{
    const float t = alpha * v + beta;
    return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}
/*@end*/

/*@kernel hard_sigmoid_f32 requires=unary_t,hard_sigmoid_scalar*/
OX_UNARY(ox_hard_sigmoid_f32, ox_hard_sigmoid(v, p0, p1))
/*@end*/

/*@kernel hard_swish_f32 requires=unary_t,hard_sigmoid_scalar*/
OX_UNARY(ox_hard_swish_f32, v *ox_hard_sigmoid(v, 1.0f / 6.0f, 0.5f))
/*@end*/

/*@kernel exp_scalar*/
/* Single precision exponential after Cephes expf, which keeps generated code free of libm. */
static float ox_pow2(int32_t n)
{
    union {
        float f;
        uint32_t u;
    } bits;
    bits.u = (uint32_t)(n + 127) << 23;
    return bits.f;
}

static float ox_exp(float x)
{
    if (x > 88.7228394f) {
        const float huge = 0x1p127f;
        return huge * 2.0f;
    }
    if (x < -103.972084f) {
        return 0.0f;
    }
    const int32_t n = (int32_t)(1.44269504f * x + (x < 0.0f ? -0.5f : 0.5f));
    const float k = (float)n;
    const float z = x - k * 0.693359375f + k * 2.12194440e-4f;
    const float zz = z * z;
    const float poly = ((((((1.9875691500e-4f * z + 1.3981999507e-3f) * z + 8.3334519073e-3f) * z +
                           4.1665795894e-2f) *
                              z +
                          1.6666665459e-1f) *
                             z +
                         5.0000001201e-1f) *
                        zz) +
                       z + 1.0f;
    const int32_t half = n / 2;
    return poly * ox_pow2(half) * ox_pow2(n - half);
}
/*@end*/

/*@kernel exp_f32 requires=unary_t,exp_scalar*/
OX_UNARY(ox_exp_f32, ox_exp(v))
/*@end*/

/*@kernel sigmoid_f32 requires=unary_t,exp_scalar*/
OX_UNARY(ox_sigmoid_f32, 1.0f / (1.0f + ox_exp(-v)))
/*@end*/

/*@kernel tanh_scalar requires=exp_scalar*/
/* Cephes tanhf: a polynomial near zero, the exponential identity elsewhere. */
static float ox_tanh(float x)
{
    const float magnitude = x < 0.0f ? -x : x;
    float t;
    if (magnitude >= 0.625f) {
        t = 1.0f - 2.0f / (ox_exp(2.0f * magnitude) + 1.0f);
        return x < 0.0f ? -t : t;
    }
    const float z = x * x;
    t = ((((-5.70498872745e-3f * z + 2.06390887954e-2f) * z - 5.37397155531e-2f) * z +
          1.33314422036e-1f) *
             z -
         3.33332819422e-1f) *
            z * x +
        x;
    return t;
}
/*@end*/

/*@kernel tanh_f32 requires=unary_t,tanh_scalar*/
OX_UNARY(ox_tanh_f32, ox_tanh(v))
/*@end*/

/*@kernel softmax_f32 requires=exp_scalar*/
/* The input is viewed as (outer, n, inner) and normalized along the middle axis. */
static void ox_softmax_f32(float *y, const float *x, int32_t outer, int32_t n, int32_t inner)
{
    for (int32_t o = 0; o < outer; ++o) {
        for (int32_t j = 0; j < inner; ++j) {
            const int32_t base = o * n * inner + j;
            float top = x[base];
            float sum = 0.0f;
            for (int32_t i = 1; i < n; ++i) {
                const float value = x[base + i * inner];
                top = value > top ? value : top;
            }
            for (int32_t i = 0; i < n; ++i) {
                const float value = ox_exp(x[base + i * inner] - top);
                y[base + i * inner] = value;
                sum += value;
            }
            for (int32_t i = 0; i < n; ++i) {
                y[base + i * inner] /= sum;
            }
        }
    }
}
/*@end*/

/*@kernel fixed_point*/
#if (-1 >> 1) != -1
#error "onnxcc integer kernels require arithmetic right shift of negative values"
#endif

static int8_t ox_saturate(int32_t v)
{
    return (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
}
/*@end*/

/*@kernel requant requires=fixed_point*/
/* Multiply by mult * 2^(-31 - shift) with mult in [2^30, 2^31). A negative shift means a gain
 * above one. A constant 64-bit shift compiles to a single SMULL-based sequence on Cortex-M4,
 * where a variable 64-bit shift costs about twenty instructions, hence two rounding steps. */
static int32_t ox_requant(int32_t value, int32_t mult, int32_t shift)
{
    if (shift < 0) {
        const int32_t limit = (int32_t)1 << (30 + shift);
        value = value < -limit ? -limit : (value > limit ? limit : value);
        value = value * ((int32_t)1 << -shift);
    }
    const int64_t product = (int64_t)value * mult + ((int64_t)1 << 30);
    const int32_t high = (int32_t)(product >> 31);
    if (shift <= 0) {
        return high;
    }
    const int32_t mask = ((int32_t)1 << shift) - 1;
    const int32_t remainder = high & mask;
    const int32_t threshold = (mask >> 1) + (high < 0 ? 1 : 0);
    return (high >> shift) + (remainder > threshold ? 1 : 0);
}
/*@end*/

/*@kernel qconv2d_s8 requires=window_t,span,requant*/
/* Same scatter form as the float kernel. `acc` is one output plane of int32 accumulators, kept in
 * a scratch array sized by the compiler, because per-pixel gather loops cost twice as many
 * instructions on Cortex-M4 and defeat vectorization on hosts. */
static void ox_qconv2d_s8(int8_t *restrict y, const int8_t *restrict x, const int8_t *restrict w,
                          const int32_t *restrict b, const int32_t *restrict mult,
                          const int32_t *restrict shift, int32_t *restrict acc,
                          const ox_window_t *p, int32_t x_zp, int32_t y_zp)
{
    const int32_t plane_out = p->h_out * p->w_out;
    const int32_t plane_in = p->h_in * p->w_in;
    const int32_t cg_in = p->c_in / p->group;
    const int32_t cg_out = p->c_out / p->group;
    for (int32_t n = 0; n < p->n; ++n) {
        const int8_t *x_batch = x + n * p->c_in * plane_in;
        int8_t *y_batch = y + n * p->c_out * plane_out;
        for (int32_t c = 0; c < p->c_out; ++c) {
            const int8_t *wc = w + c * cg_in * p->k_h * p->k_w;
            const int8_t *xg = x_batch + (c / cg_out) * cg_in * plane_in;
            int8_t *yp = y_batch + c * plane_out;
            const int32_t bias = b ? b[c] : 0;
            for (int32_t i = 0; i < plane_out; ++i) {
                acc[i] = bias;
            }
            for (int32_t ic = 0; ic < cg_in; ++ic) {
                const int8_t *xp = xg + ic * plane_in;
                for (int32_t kh = 0; kh < p->k_h; ++kh) {
                    const int32_t off_h = kh * p->d_h - p->p_t;
                    int32_t oy_lo, oy_hi;
                    ox_span(off_h, p->s_h, p->h_in, p->h_out, &oy_lo, &oy_hi);
                    for (int32_t kw = 0; kw < p->k_w; ++kw) {
                        const int32_t weight = *wc++;
                        const int32_t off_w = kw * p->d_w - p->p_l;
                        int32_t ox_lo, ox_hi;
                        ox_span(off_w, p->s_w, p->w_in, p->w_out, &ox_lo, &ox_hi);
                        const int32_t count = ox_hi - ox_lo;
                        if (count <= 0 || weight == 0) {
                            continue;
                        }
                        for (int32_t oy = oy_lo; oy < oy_hi; ++oy) {
                            const int32_t iy = oy * p->s_h + off_h;
                            const int8_t *row = xp + iy * p->w_in + ox_lo * p->s_w + off_w;
                            int32_t *out = acc + oy * p->w_out + ox_lo;
                            if (p->s_w == 1) {
                                for (int32_t i = 0; i < count; ++i) {
                                    out[i] += ((int32_t)row[i] - x_zp) * weight;
                                }
                            } else {
                                for (int32_t i = 0; i < count; ++i) {
                                    out[i] += ((int32_t)row[i * p->s_w] - x_zp) * weight;
                                }
                            }
                        }
                    }
                }
            }
            for (int32_t i = 0; i < plane_out; ++i) {
                yp[i] = ox_saturate(ox_requant(acc[i], mult[c], shift[c]) + y_zp);
            }
        }
    }
}
/*@end*/

/*@kernel qdense_s8 requires=requant*/
/* `w` is stored as [n][k]: one row per output column. */
static void ox_qdense_s8(int8_t *restrict y, const int8_t *restrict a, const int8_t *restrict w,
                         const int32_t *restrict b, const int32_t *restrict mult,
                         const int32_t *restrict shift, int32_t m, int32_t k, int32_t n,
                         int32_t a_zp, int32_t y_zp)
{
    for (int32_t r = 0; r < m; ++r) {
        const int8_t *row = a + r * k;
        for (int32_t j = 0; j < n; ++j) {
            const int8_t *wj = w + j * k;
            int32_t acc = b ? b[j] : 0;
            for (int32_t i = 0; i < k; ++i) {
                acc += ((int32_t)row[i] - a_zp) * wj[i];
            }
            *y++ = ox_saturate(ox_requant(acc, mult[j], shift[j]) + y_zp);
        }
    }
}
/*@end*/

/*@kernel qbinary_t requires=bcast_t*/
typedef struct {
    int32_t a_zp, b_zp, y_zp;
    int32_t a_mult, b_mult, bits;
    int32_t mult, shift;
} ox_qbin_t;
/*@end*/

/*@kernel qadd_s8 requires=qbinary_t,fixed_point*/
/* Align both operands to a common fixed-point scale with `bits` fractional bits. */
static int8_t ox_qadd(int32_t u, int32_t v, const ox_qbin_t *q)
{
    const int32_t sum = (u - q->a_zp) * q->a_mult + (v - q->b_zp) * q->b_mult;
    const int32_t half = (int32_t)1 << (q->bits - 1);
    return ox_saturate(((sum + half) >> q->bits) + q->y_zp);
}

OX_BINARY(ox_qadd_s8, int8_t, ox_qbin_t, ox_qadd(u, v, q))
/*@end*/

/*@kernel qmul_s8 requires=qbinary_t,requant*/
static int8_t ox_qmul(int32_t u, int32_t v, const ox_qbin_t *q)
{
    const int32_t product = (u - q->a_zp) * (v - q->b_zp);
    return ox_saturate(ox_requant(product, q->mult, q->shift) + q->y_zp);
}

OX_BINARY(ox_qmul_s8, int8_t, ox_qbin_t, ox_qmul(u, v, q))
/*@end*/

/*@kernel qavgpool_s8 requires=window_t,taps,requant*/
/* Eight fractional bits survive the integer division, and `mult` undoes them. */
static void ox_qavgpool_s8(int8_t *restrict y, const int8_t *restrict x, const ox_window_t *p,
                           int32_t include_pad, int32_t x_zp, int32_t y_zp, int32_t mult,
                           int32_t shift)
{
    const int32_t planes = p->n * p->c_in;
    for (int32_t c = 0; c < planes; ++c) {
        const int8_t *xp = x + c * p->h_in * p->w_in;
        for (int32_t oy = 0; oy < p->h_out; ++oy) {
            const int32_t top = oy * p->s_h - p->p_t;
            int32_t kh_lo, kh_hi;
            ox_taps(top, p->k_h, p->d_h, p->h_in, &kh_lo, &kh_hi);
            for (int32_t ox = 0; ox < p->w_out; ++ox) {
                const int32_t left = ox * p->s_w - p->p_l;
                int32_t kw_lo, kw_hi;
                int32_t sum = 0;
                ox_taps(left, p->k_w, p->d_w, p->w_in, &kw_lo, &kw_hi);
                for (int32_t kh = kh_lo; kh < kh_hi; ++kh) {
                    const int32_t row = (top + kh * p->d_h) * p->w_in + left;
                    for (int32_t kw = kw_lo; kw < kw_hi; ++kw) {
                        sum += (int32_t)xp[row + kw * p->d_w] - x_zp;
                    }
                }
                const int32_t count =
                    include_pad ? p->k_h * p->k_w : (kh_hi - kh_lo) * (kw_hi - kw_lo);
                const int32_t scaled = sum * 256;
                const int32_t mean = (scaled + (sum < 0 ? -count / 2 : count / 2)) / count;
                *y++ = ox_saturate(ox_requant(mean, mult, shift) + y_zp);
            }
        }
    }
}
/*@end*/

/*@kernel qgap_s8 requires=requant*/
static void ox_qgap_s8(int8_t *restrict y, const int8_t *restrict x, int32_t planes, int32_t size,
                       int32_t x_zp, int32_t y_zp, int32_t mult, int32_t shift)
{
    for (int32_t c = 0; c < planes; ++c) {
        int32_t sum = 0;
        for (int32_t i = 0; i < size; ++i) {
            sum += (int32_t)x[c * size + i] - x_zp;
        }
        y[c] = ox_saturate(ox_requant(sum, mult, shift) + y_zp);
    }
}
/*@end*/

/*@kernel qconcat_s8 requires=requant*/
/* Concat with per-input rescaling. Inputs carry eight extra bits through the multiplier. */
static void ox_qconcat_s8(int8_t *restrict y, const int8_t *const *inputs, const int32_t *chunks,
                          int32_t count, int32_t blocks, const int32_t *zps, const int32_t *mults,
                          const int32_t *shifts, int32_t y_zp)
{
    for (int32_t block = 0; block < blocks; ++block) {
        for (int32_t j = 0; j < count; ++j) {
            const int8_t *x = inputs[j] + block * chunks[j];
            for (int32_t i = 0; i < chunks[j]; ++i) {
                const int32_t value = ((int32_t)x[i] - zps[j]) * 256;
                y[i] = ox_saturate(ox_requant(value, mults[j], shifts[j]) + y_zp);
            }
            y += chunks[j];
        }
    }
}
/*@end*/

/*@kernel clip_s8*/
static void ox_clip_s8(int8_t *y, const int8_t *x, int32_t n, int32_t lo, int32_t hi)
{
    for (int32_t i = 0; i < n; ++i) {
        const int32_t v = x[i];
        y[i] = (int8_t)(v < lo ? lo : (v > hi ? hi : v));
    }
}
/*@end*/

/*@kernel quantize_s8*/
/* ONNX QuantizeLinear rounds halves to even. The clamp to +/-1000 precedes the integer conversion
 * because converting an out-of-range float is undefined, and anything beyond +/-256 saturates. */
static void ox_quantize_s8(int8_t *restrict y, const float *restrict x, int32_t n, float scale,
                           int32_t zp)
{
    for (int32_t i = 0; i < n; ++i) {
        float q = x[i] / scale;
        q = q > 1000.0f ? 1000.0f : (q > -1000.0f ? q : -1000.0f);
        int32_t t = (int32_t)q;
        const float frac = q - (float)t;
        if (frac > 0.5f || (frac >= 0.5f && (t & 1))) {
            t += 1;
        } else if (frac < -0.5f || (frac <= -0.5f && (t & 1))) {
            t -= 1;
        }
        t += zp;
        y[i] = (int8_t)(t < -128 ? -128 : (t > 127 ? 127 : t));
    }
}
/*@end*/

/*@kernel dequantize_s8*/
static void ox_dequantize_s8(float *restrict y, const int8_t *restrict x, int32_t n, float scale,
                             int32_t zp)
{
    for (int32_t i = 0; i < n; ++i) {
        y[i] = (float)((int32_t)x[i] - zp) * scale;
    }
}
/*@end*/
