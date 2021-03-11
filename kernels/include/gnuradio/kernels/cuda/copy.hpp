#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gr {
namespace kernels {
namespace cuda {

struct copy_kernel : kernel_interface {
    void operator()(void* in_buffer,
                    void* out_buffer,
                    size_t num_input_items,
                    size_t num_output_items);

    void operator()(void* in_buffer, void* out_buffer, size_t num_items);

    void operator()(void* buffer, size_t num_items);
};

} // namespace cuda
} // namespace kernels
} // namespace gr
