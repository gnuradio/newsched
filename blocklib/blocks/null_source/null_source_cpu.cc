#include "null_source_cpu.hh"
#include "null_source_cpu_gen.hh"

namespace gr {
namespace blocks {

work_return_code_t null_source_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                         std::vector<block_work_output_sptr>& work_output)
{
    void* optr;

    for (size_t n = 0; n < work_output.size(); n++) {
        optr = work_output[n]->items<void>();
        auto noutput_items = work_output[n]->n_items;
        memset(optr, 0, noutput_items * d_itemsize);
        work_output[n]->n_produced = noutput_items;
    }

    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr