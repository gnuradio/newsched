#include <gnuradio/blocklib/cuda/multiply_const.hpp>

namespace gr {
namespace cuda {

extern void multiply_const_kernel_wrapper(int N, float k, const float* in, float* out);

template <class T>
work_return_code_t multiply_const<T>::work(std::vector<block_work_input>& work_input,
                                           std::vector<block_work_output>& work_output)
{
    multiply_const_kernel_wrapper(work_output[0].n_items, d_k, (const T *)work_input[0].buffer->read_ptr(), (T *)work_output[0].buffer->write_ptr());
    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}


template class multiply_const<float>;

} // namespace cuda
} // namespace gr