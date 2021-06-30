#pragma once


// This header should be moved to math in build,
// but I have no clue how to do that in meson.
// The file structure goes down several levels,
// and I don't want to fubar anything by messing
// with the source code.
#include <gnuradio/math/multiply_const.hh>

#include <cuda.h>
#include <cuda_runtime.h>

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

    std::shared_ptr<cusp::multiply_const<T>> p_kernel;
    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};


} // namespace blocks
} // namespace gr
