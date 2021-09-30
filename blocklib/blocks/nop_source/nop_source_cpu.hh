#pragma once

#include <gnuradio/blocks/nop_source.hh>

namespace gr {
namespace blocks {

class nop_source_cpu : public nop_source
{
public:
    nop_source_cpu(block_args args) : sync_block("nop_source"), nop_source(args), d_itemsize(args.itemsize) {}
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t d_itemsize;
    size_t d_nports;
};

} // namespace blocks
} // namespace gr