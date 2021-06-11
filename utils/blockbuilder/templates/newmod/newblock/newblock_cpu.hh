#pragma once

#include <gnuradio/blocks/newblock.hh>

namespace gr {
namespace blocks {

class newblock_cpu : public newblock
{
public:
    newblock_cpu(block_args args) : newblock(args) {}
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:

};

} // namespace blocks
} // namespace gr