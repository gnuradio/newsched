#include "load_cuda.hh"

#include <gnuradio/helper_cuda.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "load_cuda.cuh"

namespace gr {
namespace blocks {

load::sptr load::make_cuda(const block_args& args) { return std::make_shared<load_cuda>(args); }

load_cuda::load_cuda(block_args args) : load(args), d_itemsize(args.itemsize), d_load(args.load)

{
    load_cu::get_block_and_grid(&d_min_grid_size, &d_block_size);
    GR_LOG_INFO(_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);

    cudaStreamCreate(&d_stream);
}

work_return_code_t load_cuda::work(std::vector<block_work_input>& work_input,
                                   std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const uint8_t*>(work_input[0].items());
    auto out = static_cast<uint8_t*>(work_output[0].items());

    auto noutput_items = work_output[0].n_items;

    load_cu::exec_kernel(
        in, out, (noutput_items * d_itemsize) / d_block_size, d_block_size, d_load, d_stream);
    checkCudaErrors(cudaPeekAtLastError());
    cudaStreamSynchronize(d_stream);


    // Tell runtime system how many output items we produced.
    work_output[0].n_produced = // noutput_items;
        (((noutput_items * d_itemsize) / d_block_size) * d_block_size) / d_itemsize;
    return work_return_code_t::WORK_OK;
}
} // namespace blocks
} // namespace gr
