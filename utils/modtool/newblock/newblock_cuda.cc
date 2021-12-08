#include "newblock_cuda.hh"
#include "newblock_cuda_gen.hh"

#include <gnuradio/helper_cuda.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace gr {
namespace newmod {

extern void apply_newblock(
    const uint8_t* in, uint8_t* out, int grid_size, int block_size, cudaStream_t stream);

extern void get_block_and_grid(int* minGrid, int* minBlock);

newblock_cuda::newblock_cuda(block_args args) : newblock(args), d_itemsize(args.itemsize)

{
    get_block_and_grid(&d_min_grid_size, &d_block_size);
    GR_LOG_INFO(_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);

    cudaStreamCreate(&d_stream);
}

work_return_code_t newblock_cuda::work(std::vector<block_work_input_sptr>& work_input,
                                   std::vector<block_work_output_sptr>& work_output)
{
    // Do <+signal processing+>
    // Block specific code goes here
    cudaStreamSynchronize(d_stream);


    // Tell runtime system how many output items we produced.
    work_output[0].n_produced = // noutput_items;
        (((noutput_items * d_itemsize) / d_block_size) * d_block_size) / d_itemsize;
    return work_return_code_t::WORK_OK;
}
} // namespace newmod
} // namespace gr
