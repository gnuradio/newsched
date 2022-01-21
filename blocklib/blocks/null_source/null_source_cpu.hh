#pragma once

#include <gnuradio/blocks/null_source.hh>

namespace gr {
namespace blocks {

class null_source_cpu : public null_source
{
public:
    null_source_cpu(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

protected:
    size_t d_nports;
};

} // namespace blocks
} // namespace gr