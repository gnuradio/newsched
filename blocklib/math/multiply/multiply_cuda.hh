#pragma once

#include <gnuradio/math/multiply.hh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cusp/multiply.cuh>

namespace gr {
namespace math {

template <class T>
class multiply_cuda : public multiply<T>
{
public:
    multiply_cuda(const typename multiply<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t num_inputs;
    size_t d_vlen;

    std::shared_ptr<cusp::multiply<T>> p_kernel;
    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};


} // namespace blocks
} // namespace gr
