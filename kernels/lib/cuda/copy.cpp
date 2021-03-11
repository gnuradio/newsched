#include <gnuradio/kernels/cuda/copy.hpp>

#include "helper_cuda.h"

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace gr {
namespace kernels {
namespace cuda {

extern void apply_copy(cuFloatComplex* in,
                       cuFloatComplex* out,
                       int grid_size,
                       int block_size,
                       int load,
                       cudaStream_t stream);

extern void get_block_and_grid(int* minGrid, int* minBlock);

void copy_kernel::operator()(void* in_buffer,
                             void* out_buffer,
                             size_t num_input_items,
                             size_t num_output_items)
{
    int min_grid_size;
    int block_size;
    int batch_size = 40;
    size_t load = 20;

    // Initialize CUDA stuff.
    get_block_and_grid(&min_grid_size, &block_size);
    cudaStream_t stream;

    cudaStreamCreate(&stream);

    auto in_data = const_cast<cuFloatComplex*>(
        reinterpret_cast<const cuFloatComplex*>(in_buffer + s * batch_size));
    auto out_data = reinterpret_cast<cuFloatComplex*>(out_buffer + s * batch_size);

    apply_copy(in_data, out_data, batch_size / block_size, block_size, load, stream);
    cudaStreamSynchronize(stream);
}

void copy_kernel::operator()(void* in_buffer, void* out_buffer, size_t num_items) {}

void copy_kernel::operator()(void* buffer, size_t num_items) {}

} // namespace cuda
} // namespace kernels
} // namespace gr
