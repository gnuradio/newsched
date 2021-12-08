#pragma once

#include <gnuradio/newmod/newblock.hh>

namespace gr {
namespace newmod {

class newblock_cpu : public newblock
{
public:
    newblock_cpu(block_args args) : sync_block("newblock"), newblock(args) {}
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    // private variables here
};

} // namespace newmod
} // namespace gr