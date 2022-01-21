#pragma once

#include <gnuradio/blocks/delay.hh>
#include <mutex>

namespace gr {
namespace blocks {

class delay_cpu : public delay
{
public:
    delay_cpu(const block_args& args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;
    size_t dly() { return d_delay; }
    void set_dly(size_t d);

protected:
    size_t d_delay = 0;
    int d_delta = 0;

    std::mutex d_mutex;
};

} // namespace blocks
} // namespace gr