#include "complex_to_mag_squared_cpu.hh"
#include <volk/volk.h>

namespace gr {
namespace math {

complex_to_mag_squared::sptr complex_to_mag_squared::make_cpu(const block_args& args) { return std::make_shared<complex_to_mag_squared_cpu>(args); }

work_return_code_t complex_to_mag_squared_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto noutput_items = work_output[0].n_items;
    int noi = noutput_items * d_vlen;

    auto iptr = work_input[0].items<gr_complex>();
    auto optr = work_output[0].items<float>();

    volk_32fc_magnitude_squared_32f(optr, iptr, noi);

    produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace math
} // namespace gr