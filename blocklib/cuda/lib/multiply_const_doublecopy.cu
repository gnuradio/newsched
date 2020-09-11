#include <cuda.h>
#include <cuda_runtime.h>

__global__ 
void multiply_const_doublecopy_kernel(int n, float a, float* in, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a * in[i];
}

namespace gr {
namespace cuda {
void multiply_const_doublecopy_kernel_wrapper(int N, float k, const float* in, float* out)
{
    float *dev_x, *dev_y;


    cudaMalloc(&dev_x, N * sizeof(float));
    cudaMalloc(&dev_y, N * sizeof(float));


    cudaMemcpy(dev_x, in, N * sizeof(float), cudaMemcpyHostToDevice);

    const int nthreads = 64;
    // Perform SAXPY on 1M elements
    multiply_const_doublecopy_kernel<<<(N + nthreads - 1) / nthreads, nthreads>>>(N, k, dev_x, dev_y);

    cudaMemcpy(out, dev_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
}

} // namespace cuda
} // namespace gr