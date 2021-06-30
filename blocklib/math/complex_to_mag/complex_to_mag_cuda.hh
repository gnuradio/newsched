#pragma once

#include <gnuradio/math/complex_to_mag.hh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cusp/complex_to_mag.cuh>

namespace gr {
namespace math {

class complex_to_mag_cuda : public complex_to_mag
{
public:
    complex_to_mag_cuda(const typename complex_to_mag::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t d_vlen;

    std::shared_ptr<cusp::complex_to_mag> p_kernel;
    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};


} // namespace blocks
} // namespace gr
