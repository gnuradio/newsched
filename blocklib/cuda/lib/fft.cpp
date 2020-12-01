#include <gnuradio/blocklib/cuda/fft.hpp>

#include "helper_cuda.h"

extern void apply_window(cuFloatComplex* in,
                         cuFloatComplex* out,
                         int fft_size,
                         int batch_size);

namespace gr {
namespace cuda {


/*
 * The private constructor
 */
fft::fft(const size_t fft_size,
         const bool forward,
         bool shift,
         const size_t batch_size)
    : gr::sync_block("fft"),
      d_fft_size(fft_size),
      d_forward(forward),
      d_shift(shift),
      d_batch_size(batch_size)

{

    size_t workSize;
    int fftSize = d_fft_size;

    checkCudaErrors(cufftCreate(&d_plan));

    checkCudaErrors(cufftMakePlanMany(
        d_plan, 1, &fftSize, NULL, 1, 1, NULL, 1, 1, CUFFT_C2C, d_batch_size, &workSize));

    gr_log_info(_logger, "Temporary buffer size {} bytes", workSize);

    // checkCudaErrors(cufftPlan1d(&d_plan, d_fft_size, CUFFT_C2C, 1));

    set_output_multiple(d_batch_size);
}

/*
 * Our virtual destructor.
 */
fft::~fft()
{
    cufftDestroy(d_plan);
}

work_return_code_t fft::work(std::vector<block_work_input>& work_input,
                             std::vector<block_work_output>& work_output)
{
    const gr_complex* in = reinterpret_cast<const gr_complex*>(work_input[0].items);
    // gr_complex* in = reinterpret_cast<gr_complex*>(work_input[0].items);
    gr_complex* out = reinterpret_cast<gr_complex*>(work_output[0].items);

    auto noutput_items = work_output[0].n_items;

    auto work_size = d_batch_size * d_fft_size;
    auto nvecs = noutput_items / d_batch_size;
    auto mem_size = work_size * sizeof(gr_complex);


    for (auto s = 0; s < nvecs; s++) {

        auto in_data = const_cast<cufftComplex*>(
            reinterpret_cast<const cufftComplex*>(in + s * work_size));
        auto out_data = reinterpret_cast<cufftComplex*>(out + s * work_size);

        if (d_forward) {
            checkCudaErrors(
            cufftExecC2C(d_plan, in_data, out_data, CUFFT_FORWARD));
        } else {
            checkCudaErrors(
            cufftExecC2C(d_plan, in_data, out_data, CUFFT_INVERSE));
        }

        cudaDeviceSynchronize();


    }

    // Tell runtime system how many output items we produced.
    work_output[0].n_produced = noutput_items;
    return work_return_code_t::WORK_OK;
}
} // namespace cuda
} // namespace gr
