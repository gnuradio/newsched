#pragma once


// This header should be moved to math in build,
// but I have no clue how to do that in meson.
// The file structure goes down several levels,
// and I don't want to fubar anything by messing
// with the source code.
#include <gnuradio/math/conjugate.hh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cusp/conjugate.cuh>

namespace gr {
namespace math {

class conjugate_cuda : public conjugate
{
public:
    conjugate_cuda(const typename conjugate::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:

    std::shared_ptr<cusp::conjugate> p_kernel;
    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};


} // namespace blocks
} // namespace gr
