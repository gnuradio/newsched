#pragma once

#include <gnuradio/filter/pfb_channelizer.hh>
#include <cusp/channelizer.cuh>
#include <cusp/deinterleave.cuh>

namespace gr {
namespace filter {

template <class T>
class pfb_channelizer_cuda : public pfb_channelizer<T>
{
public:
    pfb_channelizer_cuda(const typename pfb_channelizer<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;


private:
  size_t d_nchans;
  size_t d_taps;
  size_t d_overlap;

  void *d_dev_buf;
  void *d_dev_tail;
  cudaStream_t d_stream;

  std::shared_ptr<cusp::channelizer<gr_complex>> p_channelizer;
  std::shared_ptr<cusp::deinterleave> p_deinterleaver;

  std::vector<const void *> d_in_items;
  std::vector<void *> d_out_items;
};


} // namespace filter
} // namespace gr
