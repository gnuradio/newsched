#pragma once

#include <gnuradio/math/conjugate.hh>
#include <volk/volk.h>

namespace gr {
namespace math {

class conjugate_cpu : public conjugate
{
public:
    conjugate_cpu(const block_args& args) : sync_block("conjugate"), conjugate(args)
    {
        // const int alignment_multiple = volk_get_alignment() / sizeof(gr_complex);
        // set_output_multiple(std::max(1, alignment_multiple));
    }
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;
};

} // namespace math
} // namespace gr