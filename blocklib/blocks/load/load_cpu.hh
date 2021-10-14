#pragma once

#include <gnuradio/blocks/load.hh>

namespace gr {
namespace blocks {

class load_cpu : public load
{
public:
    load_cpu(block_args args) : sync_block("load"), load(args), d_itemsize(args.itemsize), d_load(args.iterations) {}
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t d_itemsize;
    size_t d_load;
};

} // namespace blocks
} // namespace gr