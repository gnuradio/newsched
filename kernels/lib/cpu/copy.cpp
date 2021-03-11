#include <gnuradio/kernels/cpu/copy.hpp>
#include <cstring>

namespace gr {
namespace kernels {
namespace cpu {
// Instead of template specializations, we're hoping to pick the kernel based on the port
// type, block name, and device However, the port is currently instantiated after the
// block is and even if it's not we do not have a way to enforce that the port should be
// instantiated first. At the very least, this still allows for the separation of a kernel
// library from a block/scheduler library.

// We will need template specializations anyway as we'll want to create many kernels and
// register them with the kernel registry
template <class T>
void copy_kernel<T>::operator()(void* in_buffer,
                                void* out_buffer,
                                size_t num_input_items,
                                size_t num_output_items)
{
    memcpy(out_buffer, in_buffer, num_input_items * sizeof(T));
}

template <class T>
void copy_kernel<T>::operator()(void* in_buffer, void* out_buffer, size_t num_items)
{
    memcpy(out_buffer, in_buffer, num_items * sizeof(T));
}

template <class T>
void copy_kernel<T>::operator()(void* buffer, size_t num_items)
{
}
} // namespace cpu
} // namespace kernels
} // namespace gr

// So after I templatize these kernels, I want to register them with a kernel registry.
// The kernel registry can then be queried by blocks/ports to determine qualification to
// be used in a given circumstance.
template class gr::kernels::cpu::copy_kernel<uint8_t>;
template class gr::kernels::cpu::copy_kernel<uint16_t>;
template class gr::kernels::cpu::copy_kernel<uint32_t>;
template class gr::kernels::cpu::copy_kernel<uint64_t>;
