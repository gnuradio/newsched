#pragma once

#include <gnuradio/filter/dc_blocker.hh>
#include <gnuradio/filter/moving_averager.hh>

namespace gr {
namespace filter {

template <class T>
class dc_blocker_cpu : public dc_blocker<T>
{
public:
    dc_blocker_cpu(const typename dc_blocker<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

    int group_delay();

protected:
    int d_length;
    bool d_long_form;
    kernel::moving_averager<T> d_ma_0;
    kernel::moving_averager<T> d_ma_1;
    std::unique_ptr<kernel::moving_averager<T>> d_ma_2;
    std::unique_ptr<kernel::moving_averager<T>> d_ma_3;
    std::deque<T> d_delay_line;
};


} // namespace filter
} // namespace gr
