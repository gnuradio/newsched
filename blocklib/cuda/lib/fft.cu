#include <cuComplex.h>

__global__ void
apply_window_kernel(cuFloatComplex* in, cuFloatComplex* out, float* window, int fft_size, int batch_size)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = fft_size * batch_size;
    if (i < n) {
        // int w = i % fft_size;
        out[i].x = in[i].x * window[i];
        out[i].y = in[i].y * window[i];
    }
}


void apply_window(cuFloatComplex* in, cuFloatComplex* out, float* window, int fft_size, int batch_size)
{
    apply_window_kernel<<<batch_size, fft_size>>>(in, out, window, fft_size, batch_size);
}