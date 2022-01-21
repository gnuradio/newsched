#pragma once

#include <gnuradio/blocks/head.hh>

namespace gr {
namespace blocks {

class head_cpu : public head
{
public:
    head_cpu(const block_args& args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    size_t d_nitems;
    size_t d_ncopied_items = 0;
};

} // namespace blocks
} // namespace gr