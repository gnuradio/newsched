#include "conjugate_cpu.hh"
#include <volk/volk.h>

namespace gr {
namespace math {

conjugate::sptr conjugate::make_cpu(const block_args& args) { return std::make_shared<conjugate_cpu>(args); }

work_return_code_t conjugate_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto noutput_items = work_output[0].n_items;

    auto iptr = work_input[0].items<gr_complex>();
    auto optr = work_output[0].items<gr_complex>();

    volk_32fc_conjugate_32fc(optr, iptr, noutput_items);

    produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace math
} // namespace gr