#pragma once

#include <gnuradio/math/multiply_const.hh>

#include <cusp/multiply_const.cuh>

namespace gr {
namespace math {

template <class T>
class multiply_const_cuda : public multiply_const<T>
{
public:
    multiply_const_cuda(const typename multiply_const<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    T d_k;
    size_t d_vlen;

    cudaStream_t d_stream;
    std::shared_ptr<cusp::multiply_const<T>> p_kernel;
};


} // namespace math
} // namespace gr
