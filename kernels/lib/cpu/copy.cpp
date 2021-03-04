#include <gnuradio/kernels/cpu/copy.hpp>
#include <cstring>

namespace gr {
namespace kernels {

// Instead of template specializations, we're hoping to pick the kernel based on the port
// type, block name, and device However, the port is currently instantiated after the
// block is and even if it's not we do not have a way to enforce that the port should be
// instantiated first. At the very least, this still allows for the separation of a kernel
// library from a block/scheduler library.
template <class InputType, class OutputType>
void copy_kernel<InputType, OutputType>::operator()(InputType* in_buffer,
                                                    OutputType* out_buffer,
                                                    size_t num_input_items,
                                                    size_t num_output_items)
{
    memcpy(out_buffer, in_buffer, num_input_items);
}

template <class InputType, class OutputType>
void copy_kernel<InputType, OutputType>::operator()(InputType* in_buffer,
                                                    InputType* out_buffer,
                                                    size_t num_items)
{
    memcpy(out_buffer, in_buffer, num_items);
}

template <class InputType, class OutputType>
void copy_kernel<InputType, OutputType>::operator()(InputType* buffer, size_t num_items)
{
}

} // namespace kernels
} // namespace gr
