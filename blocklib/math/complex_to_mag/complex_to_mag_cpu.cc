#include "complex_to_mag_cpu.hh"
#include "complex_to_mag_cpu_gen.hh"
#include <volk/volk.h>

namespace gr {
namespace math {

complex_to_mag_cpu::complex_to_mag_cpu(const block_args& args) : INHERITED_CONSTRUCTORS, d_vlen(args.vlen)
{
    // const int alignment_multiple = volk_get_alignment() / sizeof(float);
    // set_output_multiple(std::max(1, alignment_multiple));
}

work_return_code_t complex_to_mag_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                  std::vector<block_work_output_sptr>& work_output)
{
    auto noutput_items = work_output[0]->n_items;
    int noi = noutput_items * d_vlen;

    auto iptr = work_input[0]->items<gr_complex>();
    auto optr = work_output[0]->items<float>();

    volk_32fc_magnitude_32f_u(optr, iptr, noi);

    produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace math
} // namespace gr