#include <cuda.h>

__global__ void blockadaptN_kernel(const float *input, float *filtered, float* taps, float *train, int ntaps, int nsamps)
{
    // one thread per sample
    __shared__ float e[1024];
    __shared__ float f;

    static const double BETA = 0.00005; // FIXME figure out what this ought to be
    // FIXME add gear-shifting
    int samp_idx = threadIdx.x;


    e[samp_idx] = input[samp_idx] * (filtered[samp_idx] - train[samp_idx]);

    __syncthreads();
    // dot product of e with input
    if (samp_idx == 0) {
        f = 0;
        for (int i = 0; i < nsamps; i++) {
            f += e[i]; 
            // parallel reduction?
        }
    }

    __syncthreads();

    if (samp_idx < ntaps)
        return;
    
    // actually tap_idx here
    taps[samp_idx] -= input[samp_idx] * BETA * f;
}

__global__ void adaptN_kernel(const float* in, float *out, float* taps, float *train, int ntaps, int nsamps)
{
    __shared__ float temp[1024];
    __shared__ float e;

    static const double BETA = 0.00005; // FIXME figure out what this ought to be
    // FIXME add gear-shifting

    // Filters one output sample across N taps
    int tap_idx =  threadIdx.x;
    if (tap_idx >= ntaps)
        return;

    for (int j=0; j<nsamps; j++)
    {
        __syncthreads();
        temp[tap_idx] = taps[tap_idx] * in[j+tap_idx];

        __syncthreads();

        if (tap_idx == 0) {
            float sum = 0;
            for (int i = 0; i < ntaps; i++) {
                sum += temp[i];
            }

            e = sum - train[j];
            out[j] = sum;
        }

        __syncthreads();

        // FIXME/FML why doesn't this line work - is there something CUDA I'm missing
        taps[tap_idx] -= in[j+tap_idx] * BETA * e;
    }
}

void exec_adaptN(
    const float* in, float *out, float* taps, float* train, int ntaps, int nsamps, cudaStream_t stream)
{
    adaptN_kernel<<<1, ntaps, 0, stream>>>(in, out, taps, train, ntaps, nsamps);
}

void exec_blockadaptN(
    const float* in, float *filtered, float* taps, float* train, int ntaps, int nsamps, cudaStream_t stream)
{
    blockadaptN_kernel<<<1, nsamps, 0, stream>>>(in, filtered, taps, train, ntaps, nsamps);
}