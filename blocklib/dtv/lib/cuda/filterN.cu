#include <cuda.h>

__global__ void filterN_kernel(const float* in, float* out, float* taps, int ntaps, int nsamps)
{
    __shared__ float temp[1024];
    // Filters one output sample across N taps
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int sample_idx = idx / ntaps;
    int tap_idx = idx % ntaps;

    if (idx >= ntaps * nsamps)
        return;

    temp[threadIdx.x] = taps[tap_idx] * in[sample_idx+tap_idx];

    __syncthreads();

    if (tap_idx == 0) {
        float sum = 0;
        for (int i = 0; i < ntaps; i++) {
            sum += temp[threadIdx.x + i];
        }

        out[sample_idx] = sum;
    }
}

void exec_filterN(
    const float* in, float* out, float* taps, int ntaps, int nsamps, cudaStream_t stream)
{
    int nblocks = (ntaps * nsamps) / 1024;

    if (nblocks * 1024 != ntaps * nsamps) {
        nblocks += 1;
    }

    filterN_kernel<<<nblocks, 1024, 0, stream>>>(in, out, taps, ntaps, nsamps);
}