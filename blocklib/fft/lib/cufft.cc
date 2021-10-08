#include <gnuradio/fft/cufft.hh>
#include <cufft.h>

namespace gr {
namespace fft {

template <typename T, bool forward>
cufft<T, forward>::cufft(size_t fft_size, size_t batch_size, cudaStream_t stream) : d_fft_size(fft_size), d_batch_size(batch_size), d_stream(stream)
{
    d_logger = gr::logging::get_logger("cufft", "default");
    d_debug_logger = gr::logging::get_logger("cufft(dbg)", "debug");

    if (d_batch_size)
    {
        if (cufftPlan1d(&d_plan, d_fft_size, CUFFT_C2C, d_batch_size) != CUFFT_SUCCESS) {
            GR_LOG_ERROR(d_logger, "CUFFT error: d_plan creation failed");
            return;
        }
    }
    if (d_stream)
    {
        if ( cufftSetStream(d_plan, d_stream) != CUFFT_SUCCESS)
        {
        GR_LOG_ERROR(d_logger, "CUFFT error: Stream Association failed");
        return;
        }
    }
}

template <>
void cufft<gr_complex, true>::execute(const gr_complex* in, gr_complex* out)
{
    cufftHandle plan;
    if (!d_batch_size) {
        if (d_plan_cache.count(N) > 0) {
            plan = d_plan_cache[N];
        } else {
            checkCudaErrors(cufftPlan1d(&plan, d_fft_size, CUFFT_C2C, N));
            checkCudaErrors(cufftSetStream(plan, stream));
            d_plan_cache[N] = _plan;
            plan = _plan;
        }
    } else {
        plan = d_plan;
    }

    if (cufftExecC2C(d_plan, (cufftComplex *)in, (cufftComplex *)out, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        GR_LOG_ERROR(d_logger, "CUFFT Error: Failed to execute d_plan");
        return;
    }
}

template <>
void cufft<gr_complex, false>::execute(const gr_complex* in, gr_complex* out)
{
    if (cufftExecC2C(d_plan, (cufftComplex *)in, (cufftComplex *)out, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        GR_LOG_ERROR(d_logger, "CUFFT Error: Failed to execute d_plan");
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