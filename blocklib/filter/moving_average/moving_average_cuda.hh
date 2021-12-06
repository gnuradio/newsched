#pragma once

#include <gnuradio/filter/moving_average.hh>

#include <vector>

#include <cusp/convolve.cuh>

namespace gr {
namespace filter {

template <class T>
class moving_average_cuda : public moving_average<T>
{
public:
    moving_average_cuda(const typename moving_average<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

    int group_delay();

protected:
    size_t d_length;
    T d_scale;
    size_t d_max_iter;
    size_t d_vlen;

    size_t d_new_length;
    T d_new_scale;
    bool d_updated = false;

    std::shared_ptr<cusp::convolve<T,T>> p_kernel_full;
    std::shared_ptr<cusp::convolve<T,T>> p_kernel_valid;

};


} // namespace filter
} // namespace gr
