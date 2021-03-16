#include <cuda.h>

__global__ void deinterleave_kernel(float* in, float* out)
{

    __shared__ float tmp[828 * 12];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int e = i / 828;
    int k = i % 828;
    int v;
    if (e < 4)
        v = 4 + e + 12 * (k + (k + 138) / 207);
    else if (e < 8)
        v = 4 + e + 12 * (k + (k + 69) / 207);
    else
        v = 4 + e + 12 * (k + (k + 0) / 207);

    if (i < 12 * 832) {
        tmp[e * 828 + k] = in[v];
        __syncthreads();
        out[i] = tmp[i];
    }
}

void exec_deinterleave_kernel(float* in, float* out, cudaStream_t stream)
{
    deinterleave_kernel<<<10, 1024, 0, stream>>>(in, out);
}

__global__ void viterbi_kernel(float* in,
                               unsigned char* out,
                               float* path_metrics,
                               unsigned long long* traceback,
                               int* post_coder_state)
{
    int encoder = threadIdx.x;
    if (encoder >= 12)
        return;

    // float d_path_metrics[2][4];
    float* d_path_metrics = &path_metrics[encoder * 4 * 2];
    // unsigned long long d_traceback[2][4];
    unsigned long long* d_traceback = &traceback[encoder * 4 * 2];
    unsigned char d_phase = 0;
    // int d_post_coder_state = 0;
    float d_best_state_metric = 100000;

    int* d_post_coder_state = &post_coder_state[encoder];

    // for (unsigned int i = 0; i < 2; i++)
    //     for (unsigned int j = 0; j < 4; j++) {
    //         d_path_metrics[i][j] = 0;
    //         d_traceback[i][j] = 0;
    //     }


    const int d_was_sent[4][4] = {
        { 0, 2, 4, 6 },
        { 0, 2, 4, 6 },
        { 1, 3, 5, 7 },
        { 1, 3, 5, 7 },
    };

    /* transition_table is a table of what state we were in
       given current state and bit pair sent [state][pair] */
    const int d_transition_table[4][4] = {
        { 0, 2, 0, 2 },
        { 2, 0, 2, 0 },
        { 1, 3, 1, 3 },
        { 3, 1, 3, 1 },
    };

    int nsymbols = 828;
    // int nsymbols = 0;
    for (int k = 0; k < nsymbols; k++) {
        unsigned int best_state = 0;
        // float best_state_metric = 100000;
        d_best_state_metric = 100000;

        float input = in[encoder * nsymbols + k];
        /* Precompute distances from input to each possible symbol */
        float distances[8] = { fabsf(input + 7), fabsf(input + 5), fabsf(input + 3),
                               fabsf(input + 1), fabsf(input - 1), fabsf(input - 3),
                               fabsf(input - 5), fabsf(input - 7) };

        /* We start by iterating over all possible states */
        for (unsigned int state = 0; state < 4; state++) {
            /* Next we find the most probable path from the previous
               states to the state we are testing, we only need to look at
               the 4 paths that can be taken given the 2-bit input */
            int min_metric_symb = 0;
            float min_metric = distances[d_was_sent[state][0]] +
                               d_path_metrics[d_phase * 4 + d_transition_table[state][0]];

            for (unsigned int symbol_sent = 1; symbol_sent < 4; symbol_sent++)
                if ((distances[d_was_sent[state][symbol_sent]] +
                     d_path_metrics[d_phase * 4 +
                                    d_transition_table[state][symbol_sent]]) <
                    min_metric) {
                    min_metric = distances[d_was_sent[state][symbol_sent]] +
                                 d_path_metrics[d_phase * 4 +
                                                d_transition_table[state][symbol_sent]];
                    min_metric_symb = symbol_sent;
                }

            d_path_metrics[(d_phase ^ 1) * 4 + state] = min_metric;
            d_traceback[(d_phase ^ 1) * 4 + state] =
                (((unsigned long long)min_metric_symb) << 62) |
                (d_traceback[d_phase * 4 + d_transition_table[state][min_metric_symb]] >>
                 2);

            /* If this is the most probable state so far remember it, this
               only needs to be checked when we are about to output a path
               so this test can be saved till later if needed, if performed
               later it could also be optimized with SIMD instructions.
               Even better this check could be eliminated as we are
               outputting the tail of our traceback not the head, for any
               head state path will tend towards the optimal path with a
               probability approaching 1 in just 8 or so transitions
            */
            if (min_metric <= d_best_state_metric) {
                d_best_state_metric = min_metric;
                best_state = state;
            }
        }

        if (d_best_state_metric > 10000) {
            for (unsigned int state = 0; state < 4; state++)
                d_path_metrics[(d_phase ^ 1) * 4 + state] -= d_best_state_metric;
        }
        d_phase ^= 1;

        int y2 = (0x2 & d_traceback[d_phase * 4 + best_state]) >> 1;
        int x2 = y2 ^ *d_post_coder_state;
        *d_post_coder_state = y2;

        char r = (x2 << 1) | (0x1 & d_traceback[d_phase * 4 + best_state]);
        out[encoder * (nsymbols + 797) + 797 + k] = r;
    }
}

__global__ void viterbi_kernel2(float* in,
                                unsigned char* out,
                                float* path_metrics,
                                unsigned long long* traceback,
                                int* post_coder_state)
{

#if 1
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // if (idx >= 48)
    //     return;

    // int encoder = idx / 4;
    // int state = idx % 4;

    int encoder = blockIdx.x;
    int state = threadIdx.x;


#else
    int encoder = threadIdx.x;
    if (encoder >= 12)
        return;

#endif
    // float d_path_metrics[2][4];

    __shared__ float path_metrics_shared[12 * 2 * 4];
    __shared__ unsigned long long traceback_shared[12 * 2 * 4];
    __shared__ float min_metric_shared[12 * 4];

    // __shared__ int post_coder_state_shared[12];

    // if (state == 0) {
    for (int j = 0; j < 2; j++) {
        // for (int state = 0; state < 4; state++)
        {
            path_metrics_shared[encoder * 8 + j * 4 + state] =
                path_metrics[encoder * 8 + j * 4 + state];
            traceback_shared[encoder * 8 + j * 4 + state] =
                traceback[encoder * 8 + j * 4 + state];
        }
    }
    // }
    __syncthreads();

    // float* d_path_metrics = &path_metrics[encoder * 4 * 2];
    float* d_path_metrics = &path_metrics_shared[encoder * 4 * 2];

    // unsigned long long* d_traceback = &traceback[encoder * 4 * 2];
    unsigned long long* d_traceback = &traceback_shared[encoder * 4 * 2];
    unsigned char d_phase = 0;
    // int d_post_coder_state = 0;
    float d_best_state_metric = 100000;

    int* d_post_coder_state = &post_coder_state[encoder];
    // int* d_post_coder_state = &post_coder_state_shared[encoder];

    const int d_was_sent[4][4] = {
        { 0, 2, 4, 6 },
        { 0, 2, 4, 6 },
        { 1, 3, 5, 7 },
        { 1, 3, 5, 7 },
    };

    /* transition_table is a table of what state we were in
    given current state and bit pair sent [state][pair] */
    const int d_transition_table[4][4] = {
        { 0, 2, 0, 2 },
        { 2, 0, 2, 0 },
        { 1, 3, 1, 3 },
        { 3, 1, 3, 1 },
    };

    int nsymbols = 828;
    // int nsymbols = 0;
    for (int k = 0; k < nsymbols; k++) {
        unsigned int best_state = 0;
        // float best_state_metric = 100000;


        float input = in[encoder * nsymbols + k];
        /* Precompute distances from input to each possible symbol */
        float distances[8] = { fabsf(input + 7), fabsf(input + 5), fabsf(input + 3),
                               fabsf(input + 1), fabsf(input - 1), fabsf(input - 3),
                               fabsf(input - 5), fabsf(input - 7) };


        // /* We start by iterating over all possible states */
        // for (unsigned int state = 0; state < 4; state++)
        {


            /* Next we find the most probable path from the previous
            states to the state we are testing, we only need to look at
            the 4 paths that can be taken given the 2-bit input */
            int min_metric_symb = 0;
            float min_metric = 0;
            for (unsigned int symbol_sent = 0; symbol_sent < 4; symbol_sent++) {
                float metric = (distances[d_was_sent[state][symbol_sent]] +
                                d_path_metrics[d_phase * 4 +
                                               d_transition_table[state][symbol_sent]]);
                if (symbol_sent == 0 || metric < min_metric) {
                    min_metric = metric;
                    min_metric_symb = symbol_sent;
                }
            }

            min_metric_shared[encoder * 4 + state] = min_metric;

            // __syncthreads();

            d_path_metrics[(d_phase ^ 1) * 4 + state] = min_metric;
            d_traceback[(d_phase ^ 1) * 4 + state] =
                (((unsigned long long)min_metric_symb) << 62) |
                (d_traceback[d_phase * 4 + d_transition_table[state][min_metric_symb]] >>
                 2);

            // if (min_metric <= d_best_state_metric) {
            //     d_best_state_metric = min_metric;
            //     best_state = state;
            // }
        }

        __syncthreads();

        // if (0) //(state == 0)
        {
            d_best_state_metric = 100000;
            for (int s = 0; s < 4; s++) {
                if (min_metric_shared[encoder * 4 + s] <= d_best_state_metric) {
                    d_best_state_metric = min_metric_shared[encoder * 4 + s];
                    best_state = s;
                }
            }

            if (d_best_state_metric > 10000) {
                for (unsigned int s = 0; s < 4; s++)
                    d_path_metrics[(d_phase ^ 1) * 4 + s] -= d_best_state_metric;
            }
            d_phase ^= 1;

            int y2 = (0x2 & d_traceback[d_phase * 4 + best_state]) >> 1;
            int x2 = y2 ^ *d_post_coder_state;
            *d_post_coder_state = y2;

            char r = (x2 << 1) | (0x1 & d_traceback[d_phase * 4 + best_state]);
            out[encoder * (nsymbols + 797) + 797 + k] = r;
        }
        __syncthreads();
        // out[encoder * (nsymbols + 797) + 797 + k] = encoder;
    }
    // __syncthreads();

    // if (state == 0) {
    for (int j = 0; j < 2; j++) {
        // for (int state = 0; state < 4; state++)
        {
            path_metrics[encoder * 8 + j * 4 + state] =
                path_metrics_shared[encoder * 8 + j * 4 + state];
            traceback[encoder * 8 + j * 4 + state] =
                traceback_shared[encoder * 8 + j * 4 + state];
        }
    }
    // }
}


void exec_viterbi_kernel(float* in,
                         unsigned char* out,
                         float* path_metrics,
                         unsigned long long* traceback,
                         int* post_coder_state,
                         cudaStream_t stream)
{
    // 12 coders, 4 states - 1 block of 48 threads??
    // viterbi_kernel<<<12, 4>>>(in, out, best_state);
    viterbi_kernel2<<<12, 4, 0, stream>>>(
        in, out, path_metrics, traceback, post_coder_state);
}

__global__ void interleave_kernel(unsigned char* in, unsigned char* out)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx > 828 * 12)
        return;
    int e = idx / 828;
    int k = idx % 828;
    // int byteidx = threadIdx.x;
    // int byteidx = idx % 207;

    __shared__ unsigned char unpacked[828 * 12];

    // for (int i = 0; i < 4; i++) {
    // int k = byteidx * 4 + i;

    int dbwhere;
    if (e < 4) {
        dbwhere = 6 + e * 8 + (-2 * k) + (104 * (k / 4)) +
                  64 * ((k >= 72) + (k >= 276) + (k >= 484) + (k >= 692)) +
                  -32 * ((k >= 140) + (k >= 208) + (k >= 348) + (k >= 416) + (k >= 552) +
                         (k >= 624) + (k >= 760));
    } else if (e < 8) {
        dbwhere = 6 + e * 8 + (-2 * k) + (104 * (k / 4)) +
                  64 * ((k >= 140) + (k >= 348) + (k >= 552) + (k >= 760)) +
                  -32 * ((k >= 72) + (k >= 208) + (k >= 276) + (k >= 416) + (k >= 484) +
                         (k >= 624) + (k >= 692));
    } else {
        dbwhere = 6 + e * 8 + (-2 * k) + (104 * (k / 4)) +
                  64 * ((k >= 208) + (k >= 416) + (k >= 624)) +
                  -32 * ((k >= 72) + (k >= 140) + (k >= 276) + (k >= 348) + (k >= 484) +
                         (k >= 552) + (k >= 692) + (k >= 760));
    }

    int unpackedidx = (dbwhere >> 3) * 4 + (dbwhere & 0x7) / 2;

    // no fifo
    // unpacked[unpackedidx] = in[e * (828 + 797) + 797 + k];
    // fifo
    unpacked[unpackedidx] = in[e * (828 + 797) + k];

    // }
    __syncthreads();

    // progress the fifo
    if (k < 797) {
        in[e * (828 + 797) + k] = in[e * (828 + 797) + k + 828];
    }

    if ((k % 4) != 3) // the lookup table sorts as 3,2,1,0
        return;

    // Now, pack the dibits into bytes

    out[unpackedidx / 4] = ((unpacked[unpackedidx + 3] & 0x03) << 6) |
                           ((unpacked[unpackedidx + 2] & 0x03) << 4) |
                           ((unpacked[unpackedidx + 1] & 0x03) << 2) |
                           ((unpacked[unpackedidx + 0] & 0x03) << 0);

    // out[unpackedidx/4] = unpackedidx;
}

void exec_interleave_kernel(unsigned char* in, unsigned char* out, cudaStream_t stream)
{
    interleave_kernel<<<10, 1024, 0, stream>>>(in, out); // need 828*12 threads
}
