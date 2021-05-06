#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void
multiply_kernel_ccc(cuFloatComplex* in, cuFloatComplex* out, cuFloatComplex* a, int veclen, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = veclen * batch_size;

    // int which_batch = i / veclen;
    int batch_idx = i % veclen;

    if (i < n) {
        float re, im;
        re = in[i].x * a[batch_idx].x - in[i].y * a[batch_idx].y;
        im = in[i].x * a[batch_idx].y + in[i].y * a[batch_idx].x;
        out[i].x = re;
        out[i].y = im;
    }
}

void exec_multiply_kernel_ccc(cuFloatComplex* in, cuFloatComplex* out, cuFloatComplex* a, int veclen, int batch_size,
    cudaStream_t stream)
{
    int block_size = 1024; // max num of threads
    int nblocks = (veclen * batch_size + block_size - 1) / block_size;
    multiply_kernel_ccc<<<nblocks, block_size, 0, stream>>>(in, out, a, veclen, batch_size);
}