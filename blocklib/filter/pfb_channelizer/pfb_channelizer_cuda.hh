#pragma once

#include <gnuradio/filter/pfb_channelizer.hh>
#include <gnuradio/fft/cufft.hh>


#include <mutex>

namespace gr {
namespace filter {

template <class T>
class pfb_channelizer_cuda : public pfb_channelizer<T>,
                                            kernel::polyphase_filterbank
{
public:
    pfb_channelizer_cuda(const typename pfb_channelizer<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;


private:
    void set_taps(const std::vector<float>& taps);
    cufft_complex_rev d_fft;
    std::vector<float *> d_dev_taps;
    gr_complex *d_dev_fftbuf;
    std::vector<gr_complex> d_host_fftbuf;

    size_t d_taps_per_filter;
    size_t d_nfilts;

    size_t d_history = 1;
    std::vector<std::vector<float>> d_taps;
    std::vector<std::shared_ptr<cusp::dot_product<float>>> d_filters;
};


} // namespace filter
} // namespace gr
