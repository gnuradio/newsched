#include <gnuradio/fft/cufft.hh>
#include <cufft.h>

namespace gr {
namespace fft {

template <typename T, bool forward>
cufft<T, forward>::cufft(size_t fft_size, size_t batch_size) : d_fft_size(fft_size), d_batch_size(batch_size)
{
    d_logger = gr::logging::get_logger("cufft", "default");
    d_debug_logger = gr::logging::get_logger("cufft(dbg)", "debug");
    if (cufftPlan1d(&plan, d_fft_size, CUFFT_C2C, d_batch_size) != CUFFT_SUCCESS) {
        GR_LOG_ERROR(d_logger, "CUFFT error: Plan creation failed");
        return;
    }
}

template <>
void cufft<gr_complex, true>::execute(const gr_complex* in, gr_complex* out)
{
    if (cufftExecC2C(plan, (cufftComplex *)in, (cufftComplex *)out, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        GR_LOG_ERROR(d_logger, "CUFFT Error: Failed to execute plan");
        return;
    }
}

template <>
void cufft<gr_complex, false>::execute(const gr_complex* in, gr_complex* out)
{
    if (cufftExecC2C(plan, (cufftComplex *)in, (cufftComplex *)out, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        GR_LOG_ERROR(d_logger, "CUFFT Error: Failed to execute plan");
        return;
    }
}

template class cufft<gr_complex, true>;
template class cufft<gr_complex, false>;
template class cufft<float, true>;
template class cufft<float, false>;

// template <typename T, bool forward>
// cufft<T,forward>::~cufft() {};

} // namespace fft
} // namespace gr