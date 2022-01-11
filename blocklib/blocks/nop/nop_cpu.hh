#pragma once

#include <gnuradio/blocks/nop.hh>

namespace gr {
namespace blocks {

class nop_cpu : public nop
{
public:
    nop_cpu(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

protected:
    size_t d_itemsize;
};

} // namespace blocks
} // namespace gr