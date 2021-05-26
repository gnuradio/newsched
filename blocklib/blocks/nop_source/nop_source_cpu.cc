#include "nop_source_cpu.hh"

namespace gr {
namespace blocks {

nop_source::sptr nop_source::make_cpu(const block_args& args)
{
    return std::make_shared<nop_source_cpu>(args);
}

work_return_code_t nop_source_cpu::work(std::vector<block_work_input>& work_input,
                                         std::vector<block_work_output>& work_output)
{
    void* optr;

    for (size_t n = 0; n < work_output.size(); n++) {
        optr = work_output[n].items();
        auto noutput_items = work_output[n].n_items;
        // memset(optr, 0, noutput_items * d_itemsize);
        work_output[n].n_produced = noutput_items;
    }

    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr