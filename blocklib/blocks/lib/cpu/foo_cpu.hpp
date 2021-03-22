
#pragma once

#include "../impl/foo_impl.hpp"

namespace gr {
namespace blocks {
class foo_cpu : public foo_impl
{
public:
    foo_cpu(int k) : foo_impl(k) {}

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;
};
} // namespace blocks
} // namespace gr