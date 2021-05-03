#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void
add_kernel_cc(cuFloatComplex* in, cuFloatComplex* out, cuFloatComplex* a, int veclen, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = veclen * batch_size;

    // int which_batch = i / veclen;
    int batch_idx = i % veclen;

    if (i < n) {
        float re = in[i].x + a[batch_idx].x;
        float im = in[i].y + a[batch_idx].y;
        out[i].x = re;
        out[i].y = im;
    }
}

void exec_add_kernel_cc(cuFloatComplex* in, cuFloatComplex* out, cuFloatComplex* a, int veclen, int batch_size,
    cudaStream_t stream)
{
    int block_size = 1024; // max num of threads
    int nblocks = (veclen * batch_size + block_size - 1) / block_size;
    add_kernel_cc<<<nblocks, block_size, 0, stream>>>(in, out, a, veclen, batch_size);
}


__global__ void
add_kernel_ff(float* in, float* out, float* a, int veclen, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = veclen * batch_size;

    // int which_batch = i / veclen;
    int batch_idx = i % veclen;

    if (i < n) {
        out[i] = in[i] + a[batch_idx];
    }
}

void exec_add_kernel_ff(float* in, float* out, float* a, int veclen, int batch_size,
    cudaStream_t stream)
{
    int block_size = 1024; // max num of threads
    int nblocks = (veclen * batch_size + block_size - 1) / block_size;
    add_kernel_ff<<<nblocks, block_size, 0, stream>>>(in, out, a, veclen, batch_size);
}