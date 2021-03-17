#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

// const int atsc_fs_checker_nstreams = 2;
// cudaStream_t atsc_fs_checker_streams[atsc_fs_checker_nstreams];

__global__ void atsc_fs_checker(const float* in,
                                uint8_t* pn_seq,
                                int pn_len,
                                int offset,
                                uint16_t* nerrors)
{
    int bitidx = threadIdx.x;
    int frameidx = blockIdx.x;

    if ((bitidx < offset) || (bitidx >= (offset + pn_len))) {
        return;
    }

    extern __shared__ uint8_t xored[];

    xored[bitidx - offset] = pn_seq[bitidx - offset] ^ (in[bitidx+frameidx*832] >= 0);

    __syncthreads();
    int e = 0;
    if (bitidx == offset) {
        for (int i = 0; i < pn_len; i++) {
            e += xored[i];
        }

        nerrors[frameidx] = e;
    }
}

void exec_atsc_fs_checker(const float* in,
                          uint8_t* pn_seq,
                          int pn_len,
                          int offset,
                          int nitems,
                          uint16_t* nerrors,
                          cudaStream_t str)
{

    int nthreads = 832;
    int nblocks = nitems;
    // atsc_fs_checker<<<nblocks, nthreads, pn_len, atsc_fs_checker_streams[stream_index]>>>(
    //     in, pn_seq, pn_len, offset, nerrors);
    // atsc_fs_checker<<<nblocks, nthreads, pn_len, str>>>(
    //     in, pn_seq, pn_len, offset, nerrors);
    atsc_fs_checker<<<nblocks, nthreads, pn_len, str>>>(
        in, pn_seq, pn_len, offset, nerrors);
}

