#include "copy_cuda.hh"

#include <gnuradio/helper_cuda.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace gr {
namespace blocks {

extern void apply_copy(
    const uint8_t* in, uint8_t* out, int n, int grid_size, int block_size, cudaStream_t stream);

extern void get_block_and_grid(int* minGrid, int* minBlock);

copy::sptr copy::make_cuda(const block_args& args) { return std::make_shared<copy_cuda>(args); }

copy_cuda::copy_cuda(block_args args) : sync_block("copy_cuda"), copy(args), d_itemsize(args.itemsize)

{
    get_block_and_grid(&d_min_grid_size, &d_block_size);
    GR_LOG_INFO(_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);

    cudaStreamCreate(&d_stream);
}

work_return_code_t copy_cuda::work(std::vector<block_work_input>& work_input,
                                   std::vector<block_work_output>& work_output)
{
    auto in = work_input[0].items<uint8_t>());
    auto out = work_output[0].items<uint8_t>());

    auto noutput_items = work_output[0].n_items;
    int gridSize = (noutput_items * d_itemsize + d_block_size - 1) / d_block_size;
    apply_copy(
        in, out, noutput_items * d_itemsize, gridSize, d_block_size, d_stream);
    checkCudaErrors(cudaPeekAtLastError());
    cudaStreamSynchronize(d_stream);


    // Tell runtime system how many output items we produced.
    work_output[0].n_produced = noutput_items;
    return work_return_code_t::WORK_OK;
}
} // namespace blocks
} // namespace gr
