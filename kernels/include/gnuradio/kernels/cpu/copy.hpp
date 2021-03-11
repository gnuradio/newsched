// This should live in some other folder as its dependencies should be
// quite different. In fact, it should not depend on almost anything that's currently
// in this code-base so that it can be easily included in an external library.
// That implies that kernels cannot use the std::vector<block_work_input> and
// std::vector<block_work_output> as its function signature. It should be something like
// (int *in, int *out).

// Kernels must have information about its type, device, and name (functionality) so that
// it can be matched with the correct block/port. It seems to me that the block class
// serves only the purpose of glue between ports and kernels.

#pragma once
#include <gnuradio/kernels/common/kernel.hpp>
#include <cstddef>
#include <cstdint>

namespace gr {
namespace kernels {

template <class T>
struct copy_kernel : kernel_interface {
    void operator()(void* in_buffer,
                    void* out_buffer,
                    size_t num_input_items,
                    size_t num_output_items);

    void operator()(void* in_buffer, void* out_buffer, size_t num_items);

    void operator()(void* buffer, size_t num_items);
};


} // namespace kernels
} // namespace gr
