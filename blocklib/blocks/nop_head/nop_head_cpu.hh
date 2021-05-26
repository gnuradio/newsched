#pragma once

#include <gnuradio/blocks/nop_head.hh>

namespace gr {
namespace blocks {

class nop_head_cpu : public nop_head
{
public:
    nop_head_cpu(const block_args& args);
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

private:
    size_t d_itemsize;
    size_t d_nitems;
    size_t d_ncopied_items = 0;
};

} // namespace blocks
} // namespace gr