#pragma once


// This header should be moved to math in build,
// but I have no clue how to do that in meson.
// The file structure goes down several levels,
// and I don't want to fubar anything by messing
// with the source code.
#include <gnuradio/math/complex_to_mag_squared.hh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cusp/complex_to_mag_squared.cuh>

namespace gr {
namespace math {

class complex_to_mag_squared_cuda : public complex_to_mag_squared
{
public:
    complex_to_mag_squared_cuda(const typename complex_to_mag_squared::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t d_vlen;

    std::shared_ptr<cusp::complex_to_mag_squared> p_kernel;
    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};


} // namespace blocks
} // namespace gr
