#include "copy_cpu.hpp"

namespace gr {
namespace blocks {

work_return_code_t copy_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto* iptr = (uint8_t*)work_input[0].items();
    int size = work_output[0].n_items * _itemsize;
    auto* optr = (uint8_t*)work_output[0].items();
    std::copy(iptr, iptr + size, optr);

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}

} // namespace blocks
} // namespace gr
