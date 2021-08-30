#pragma once

#include <gnuradio/fft/fft.hh>
#include <gnuradio/fft/cufft.hh>

namespace gr {
namespace fft {

template <class T, bool forward>
class fft_cuda : public fft<T,forward>
{
public:
    fft_cuda(const typename fft<T,forward>::block_args& args); 
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t d_fft_size;
    std::vector<float> d_window;
    bool d_shift;
    cudaStream_t d_stream;

    cufft<gr_complex, forward> d_fft;

    void fft_and_shift(const T* in, gr_complex* out, int batch);
};

} // namespace fft
} // namespace gr