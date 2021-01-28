__global__ 
void multiply_const_kernel(int n, float a, const float* in, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a * in[i];
}

namespace gr {
namespace cuda {
void multiply_const_kernel_wrapper(int N, float k, const float* in, float* out)
{
    const int nthreads = 64;
    // Perform SAXPY on 1M elements
    multiply_const_kernel<<<(N + nthreads - 1) / nthreads, nthreads>>>(N, k, in, out);

}

} // namespace cuda
} // namespace gr
