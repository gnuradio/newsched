#include <cuda.h>

__global__ void adaptN_kernel(float* in, float *out, float* taps, float *train, int ntaps, int nsamps)
{
    __shared__ float temp[1024];
    __shared__ float accum;
    __shared__ float e;

    static const double BETA = 0.00005; // FIXME figure out what this ought to be
    // FIXME add gear-shifting

    // Filters one output sample across N taps
    int tap_idx =  threadIdx.x;
    if (tap_idx >= 64)
        return;

    for (int j=0; j<nsamps; j++)
    {
        __syncthreads();
        temp[threadIdx.x] = taps[tap_idx] * in[j+tap_idx];

        __syncthreads();

        if (tap_idx == 0) {
            float sum = 0;
            for (int i = 0; i < ntaps; i++) {
                sum += temp[i];
            }

            accum = sum;
            e = accum - train[j];
            out[j] = accum;
        }

        __syncthreads();

        taps[tap_idx] = taps[tap_idx] - in[j+tap_idx] * BETA * e;

    }
}

void exec_adaptN(
    float* in, float *out, float* taps, float* train, int ntaps, int nsamps, cudaStream_t stream)
{
    adaptN_kernel<<<1, ntaps, 0, stream>>>(in, out, taps, train, ntaps, nsamps);
}