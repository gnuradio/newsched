#pragma once

#include <gnuradio/fft/fft.hh>
#include <gnuradio/fft/fftw_fft.hh>

namespace gr {
namespace fft {

template <class T, bool forward>
class fft_cpu : public fft<T,forward>
{
public:
    fft_cpu(const typename fft<T,forward>::block_args& args); 
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

    void set_nthreads(int n);
    int nthreads() const;

protected:
    size_t d_fft_size;
    std::vector<float> d_window;
    bool d_shift;

    fftw_fft<gr_complex, forward> d_fft;

    void fft_and_shift(const T* in, gr_complex* out);
};

} // namespace fft
} // namespace gr