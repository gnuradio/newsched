#pragma once

#include <gnuradio/filter/pfb_channelizer.hh>
#include <gnuradio/filter/polyphase_filterbank.h>

#include <mutex>

namespace gr {
namespace filter {

template <class T>
class pfb_channelizer_cpu : public pfb_channelizer<T>,
                                            kernel::polyphase_filterbank
{
public:
    pfb_channelizer_cpu(const typename pfb_channelizer<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

    int group_delay();
    void set_taps(const std::vector<float>& taps) override;

private:
    bool d_updated = false;
    float d_oversample_rate;
    std::vector<int> d_idxlut;
    int d_rate_ratio;
    int d_output_multiple;
    std::vector<int> d_channel_map;
    std::mutex d_mutex; // mutex to protect set/work access

    size_t d_history = 1;

    size_t d_nchans;
};


} // namespace filter
} // namespace gr
