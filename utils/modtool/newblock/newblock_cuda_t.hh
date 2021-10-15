#pragma once

#include <gnuradio/newmod/newblock.hh>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gr {
namespace newmod {

template <class T>
class newblock_cuda : public newblock<T>
{
public:
    newblock_cuda(const typename newblock<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    T d_k;
    size_t d_vlen;

    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};


} // namespace newmod
} // namespace gr
