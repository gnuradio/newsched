#pragma once

#include <gnuradio/blocks/throttle.hh>

namespace gr {
namespace blocks {

class throttle_cpu : public throttle
{
public:
    throttle_cpu(block_args args)
        : sync_block("throttle"), throttle(args),
          d_itemsize(args.itemsize),
          d_ignore_tags(args.ignore_tags)
    {
        set_sample_rate(args.samples_per_sec);
    }
    void set_sample_rate(double rate);

    bool start();
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    const size_t d_itemsize;
    double d_samps_per_sec;
    bool d_ignore_tags;

    std::chrono::time_point<std::chrono::steady_clock> d_start;
    uint64_t d_total_samples;
    double d_sample_rate;
    std::chrono::duration<double> d_sample_period;
};

} // namespace blocks
} // namespace gr