#pragma once

#include <gnuradio/math/conjugate.hh>

namespace gr {
namespace math {

class conjugate_cpu : public conjugate
{
public:
    conjugate_cpu(const block_args& args) : conjugate(args) {}
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

};

} // namespace math
} // namespace gr