#pragma once

#include <gnuradio/newmod/newblock.hh>

namespace gr {
namespace newmod {

class newblock_cpu : public newblock
{
public:
    newblock_cpu(block_args args) : newblock(args), d_itemsize(args.itemsize) {}
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t d_itemsize;
};

} // namespace newmod
} // namespace gr