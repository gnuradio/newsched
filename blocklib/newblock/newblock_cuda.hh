#pragma once

#include <gnuradio/newmod/newblock.hh>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gr {
namespace newmod {

class newblock_cuda : public newblock
{
public:
    newblock_cuda(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t d_itemsize;

    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};

} // namespace newmod
} // namespace gr