#pragma once

#include <gnuradio/blocks/multiply_const.hh>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gr {
namespace blocks {

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

    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};


} // namespace blocks
} // namespace gr
