#include "load_cpu.hh"

namespace gr {
namespace blocks {

load::sptr load::make_cpu(const block_args& args) { return std::make_shared<load_cpu>(args); }

work_return_code_t load_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto iptr = work_input[0].items<uint8_t>();
    auto optr = work_output[0].items<uint8_t>();
    int size = work_output[0].n_items * d_itemsize;

    // std::load(iptr, iptr + size, optr);
    for (size_t i=0; i < d_load; i++)
        memcpy(optr, iptr, size);

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr