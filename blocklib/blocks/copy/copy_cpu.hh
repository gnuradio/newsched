#pragma once

#include <gnuradio/blocks/copy.hh>

namespace gr {
namespace blocks {

class copy_cpu : public copy
{
public:
    copy_cpu(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;
};

} // namespace blocks
} // namespace gr