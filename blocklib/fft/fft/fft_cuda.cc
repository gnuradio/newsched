#include "fft_cuda.hh"

#include <gnuradio/helper_cuda.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace gr {
namespace fft {

extern void apply_fft(
    const uint8_t* in, uint8_t* out, int grid_size, int block_size, cudaStream_t stream);

extern void get_block_and_grid(int* minGrid, int* minBlock);

fft::sptr fft::make_cuda(const block_args& args) { return std::make_shared<fft_cuda>(args); }

fft_cuda::fft_cuda(block_args args) : fft(args), d_itemsize(args.itemsize)

{
    get_block_and_grid(&d_min_grid_size, &d_block_size);
    GR_LOG_INFO(_logger, "minGrid: {}, fftize: {}", d_min_grid_size, d_block_size);

    cudaStreamCreate(&d_stream);
}

work_return_code_t fft_cuda::work(std::vector<block_work_input>& work_input,
                                   std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const uint8_t*>(work_input[0].items());
    auto out = static_cast<uint8_t*>(work_output[0].items());

    auto noutput_items = work_output[0].n_items;

    apply_fft(
        in, out, (noutput_items * d_itemsize) / d_block_size, d_block_size, d_stream);
    checkCudaErrors(cudaPeekAtLastError());
    cudaStreamSynchronize(d_stream);


    // Tell runtime system how many output items we produced.
    work_output[0].n_produced = // noutput_items;
        (((noutput_items * d_itemsize) / d_block_size) * d_block_size) / d_itemsize;
    return work_return_code_t::WORK_OK;
}
} // namespace fft
} // namespace gr
