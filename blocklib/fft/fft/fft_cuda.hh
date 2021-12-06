#pragma once

#include <gnuradio/fft/fft.hh>
#include <cusp/fft.cuh>

namespace gr {
namespace fft {

template <class T, bool forward>
class fft_cuda : public fft<T,forward>
{
public:
    fft_cuda(const typename fft<T,forward>::block_args& args); 
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

protected:
    size_t d_fft_size;
    std::vector<float> d_window;
    bool d_shift;
    cudaStream_t d_stream;

    // std::shared_ptr<cusp::fft<T,forward>> d_fft;
    cusp::fft<T,forward> d_fft;

    void fft_and_shift(const T* in, gr_complex* out, int batch);
};

} // namespace fft
} // namespace gr