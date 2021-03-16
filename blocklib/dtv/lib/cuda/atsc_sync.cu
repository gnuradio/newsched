#include <cuda.h>
#include <stdint.h>

#define OUTPUT_MULTIPLE 1
#define NTAPS 8

__global__ void atsc_sync_and_integrate_kernel(
    float* in, float* interp, float* interp_taps, int8_t* integrator_accum, float* params)
{
    float ADJUSTMENT_GAIN = 1.0e-5 / (10 * 832);
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int sample_idx = idx / 8;
    int tap_idx = idx % 8;

    if (idx >= NTAPS * 832)
        return;

    __shared__ float temp[1024];
    __shared__ uint8_t bits[832 + 3];

    volatile int n;

    for (n = 0; n < OUTPUT_MULTIPLE; n++) {

        float timing_adjust = params[0];
        float mu_start = params[1];
        float w = params[2];
        int sum_si = (int)params[3];
        static const int N = NTAPS;


        float accum_value =
            mu_start + sample_idx * (ADJUSTMENT_GAIN * 1e3 * timing_adjust + w);
        int si = (int)accum_value;
        float mu = accum_value - (int)accum_value;
        int fi = (int)rint(mu * 128);

        temp[threadIdx.x] = interp_taps[8 * fi + tap_idx] * in[si+sum_si + tap_idx];

        __syncthreads();

        if (tap_idx == 0) {
            float sum = 0;
            for (int i = 0; i < N; i++) {
                sum += temp[threadIdx.x + i];
            }

            interp[sample_idx + n * 832] = sum;

            if (sample_idx == 0) { //832 - 1) {
                float accum_value =
                    mu_start + 832 * (ADJUSTMENT_GAIN * 1e3 * timing_adjust + w);
                int si = (int)accum_value;
                float mu = accum_value - (int)accum_value;
                int fi = (int)rint(mu * 128);

                params[1] = mu;
                params[3] += si;
                // sum_si += si;
            }
        }
        __syncthreads();

        // Integrator Kernel
        int integrator_idx = threadIdx.x;
        if (integrator_idx < 832 && blockIdx.x == 0) {

            // TODO - pass in 3 sample history
            // first, calculate the bits from the interpolated samples
            bits[integrator_idx + 3] = interp[integrator_idx + n * 832] < 0 ? 0 : 1;

            __syncthreads();
            int16_t sr = ((bits[integrator_idx + 3] & 0x01) << 3) |
                         ((bits[integrator_idx + 2] & 0x01) << 2) |
                         ((bits[integrator_idx + 1] & 0x01) << 1) |
                         ((bits[integrator_idx + 0] & 0x01) << 0);


            integrator_accum[integrator_idx] += ((sr == 0x9) ? +2 : -1);
            if (integrator_accum[integrator_idx] < -16)
                integrator_accum[integrator_idx] = -16;
            if (integrator_accum[integrator_idx] > 15)
                integrator_accum[integrator_idx] = 15;

            __syncthreads();

            // TODO: replace with parallel reduction
            if (integrator_idx == 0) {
                uint16_t best_idx = 0;
                int16_t best_val = integrator_accum[0];

                for (int i = 1; i < 832; i++) {
                    if (integrator_accum[i] > best_val) {
                        best_idx = i;
                        best_val = integrator_accum[i];
                    }
                }

                // *out_idx = best_idx;
                // *out_val = best_val;
                params[4] = best_idx;
                params[5] = best_val;

                int corr_count = best_idx;


                float ta = -interp[n * 832 + corr_count--];
                if (corr_count < 0)
                    corr_count = 832 - 1;
                ta -= interp[n * 832 + corr_count--];
                if (corr_count < 0)
                    corr_count = 832 - 1;
                ta += interp[n * 832 + corr_count--];
                if (corr_count < 0)
                    corr_count = 832 - 1;
                ta += interp[n * 832 + corr_count--];

                // *timing_adjust = ta;
                params[0] = ta;
            }
        }
        else
        {
            __syncthreads();
            __syncthreads();
        }
        __syncthreads();
    }

    // if (tap_idx == 0 && (sample_idx == (832 - 1))) {
    //     params[3] = sum_si;
    // }

    __syncthreads();

    // params[0] = n;
}

void exec_atsc_sync_and_integrate(float* in,
                                  float* interp,
                                  float* interp_taps,
                                  int8_t* integrator_accum,
                                  float* params,
                                  cudaStream_t stream)
{
    atsc_sync_and_integrate_kernel<<<7, 1024, 0, stream>>>(
        in, interp, interp_taps, integrator_accum, params);
}
