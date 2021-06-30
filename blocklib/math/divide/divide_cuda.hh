#pragma once


// This header should be moved to math in build,
// but I have no clue how to do that in meson.
// The file structure goes down several levels,
// and I don't want to fubar anything by messing
// with the source code.
#include <gnuradio/math/divide.hh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cusp/divide.cuh>

namespace gr {
namespace math {

template <class T>
class divide_cuda : public divide<T>
{
public:
    divide_cuda(const typename divide<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t num_inputs;
    size_t d_vlen;

    std::shared_ptr<cusp::divide<T>> p_kernel;
    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};


} // namespace blocks
} // namespace gr
