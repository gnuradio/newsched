#include "conjugate_cpu.hh"
#include "conjugate_cpu_gen.hh"
#include <volk/volk.h>

namespace gr {
namespace math {

conjugate_cpu::conjugate_cpu(const block_args& args) : INHERITED_CONSTRUCTORS
{
    // const int alignment_multiple = volk_get_alignment() / sizeof(gr_complex);
    // set_output_multiple(std::max(1, alignment_multiple));
}

work_return_code_t conjugate_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                       std::vector<block_work_output_sptr>& work_output)
{
    auto noutput_items = work_output[0]->n_items;

    auto iptr = work_input[0]->items<gr_complex>();
    auto optr = work_output[0]->items<gr_complex>();

    volk_32fc_conjugate_32fc(optr, iptr, noutput_items);

    produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace math
} // namespace gr