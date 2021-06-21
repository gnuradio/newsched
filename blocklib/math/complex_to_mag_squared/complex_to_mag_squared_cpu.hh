#pragma once

#include <gnuradio/math/complex_to_mag_squared.hh>

namespace gr {
namespace math {

class complex_to_mag_squared_cpu : public complex_to_mag_squared
{
public:
    complex_to_mag_squared_cpu(const block_args& args) : complex_to_mag_squared(args), d_vlen(args.vlen) {}
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

private:
    size_t d_vlen;

};

} // namespace math
} // namespace gr