#pragma once

#include <gnuradio/blocks/head.hh>

namespace gr {
namespace blocks {

class head_cpu : public head
{
public:
    head_cpu(size_t itemsize, size_t nitems);
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

private:
    size_t d_itemsize;
    size_t d_nitems;
    size_t d_ncopied_items = 0;
};

} // namespace blocks
} // namespace gr