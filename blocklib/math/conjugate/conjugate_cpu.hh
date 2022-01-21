#pragma once

#include <gnuradio/math/conjugate.hh>
#include <volk/volk.h>

namespace gr {
namespace math {

class conjugate_cpu : public conjugate
{
public:
    conjugate_cpu(const block_args& args);

    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;
};

} // namespace math
} // namespace gr