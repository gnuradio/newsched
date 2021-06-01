#include "copy_cuda.hh"

#include <gnuradio/helper_cuda.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cusp/copy.cuh>


namespace gr {
namespace blocks {

copy::sptr copy::make_cuda(const block_args& args) { return std::make_shared<copy_cuda>(args); }

copy_cuda::copy_cuda(block_args args) : copy(args), d_itemsize(args.itemsize)

{
    // get_block_and_grid(&d_min_grid_size, &d_block_size);

    // hardcoded for now until we can get from cusp
    d_min_grid_size = 40;
    d_block_size = 1024;

    GR_LOG_INFO(_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);

    cudaStreamCreate(&d_stream);
}

work_return_code_t copy_cuda::work(std::vector<block_work_input>& work_input,
                                   std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const uint8_t*>(work_input[0].items());
    auto out = static_cast<uint8_t*>(work_output[0].items());

    auto noutput_items = work_output[0].n_items;

    cusp::launch_kernel_copy<uint8_t>(in, out, (noutput_items * d_itemsize) / d_block_size, d_block_size,
                            noutput_items * d_itemsize, d_stream);

    checkCudaErrors(cudaPeekAtLastError());

    cudaStreamSynchronize(d_stream);


    // Tell runtime system how many output items we produced.
    work_output[0].n_produced = // noutput_items;
        (((noutput_items * d_itemsize) / d_block_size) * d_block_size) / d_itemsize;
    return work_return_code_t::WORK_OK;
}
} // namespace blocks
} // namespace gr
