/* Store adjacent tokens in each half pair so the dot consumes them directly.
 * Bounds checks surround whole copy loops to keep the B16 loads vectorized. */
const uint B_STRIDE = BN / 2u + 1u;
shared f16vec2 buf_b[BK * B_STRIDE];

void stage_token_pairs(uint lid, uint tok0, uint qb, uint slice) {
    for (uint i = lid; i < BN; i += 128u) {
        uint pair = i >> 1u, kh = i & 1u, tok = tok0 + 2u * pair;
        f16vec2 v0[8], v1[8];
        [[unroll]] for (uint j = 0u; j < 8u; j++) {
            v0[j] = f16vec2(0.0);
            v1[j] = f16vec2(0.0);
        }
        if (tok < pc.n_tok) {
            uint src = (tok * pc.blocks + qb) * 128u + slice * 16u + 8u * kh;
            [[unroll]] for (uint j = 0u; j < 8u; j++)
                v0[j] = unpackFloat2x16(b16[src + j]);
        }
        if (tok + 1u < pc.n_tok) {
            uint src = ((tok + 1u) * pc.blocks + qb) * 128u + slice * 16u + 8u * kh;
            [[unroll]] for (uint j = 0u; j < 8u; j++)
                v1[j] = unpackFloat2x16(b16[src + j]);
        }
        [[unroll]] for (uint j = 0u; j < 8u; j++) {
            uint k = 16u * kh + 2u * j;
            buf_b[k * B_STRIDE + pair] = f16vec2(v0[j].x, v1[j].x);
            buf_b[(k + 1u) * B_STRIDE + pair] = f16vec2(v0[j].y, v1[j].y);
        }
    }
}
