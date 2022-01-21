#pragma once

#include <gnuradio/blocks/copy.hh>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gr {
namespace blocks {

class copy_cuda : public copy
{
public:
    copy_cuda(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

protected:
    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};

} // namespace blocks
} // namespace gr