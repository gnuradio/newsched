#include "msg_forward_cpu.hh"

namespace gr {
namespace blocks {

msg_forward::sptr msg_forward::make_cpu(const block_args& args) { return std::make_shared<msg_forward_cpu>(args); }

work_return_code_t msg_forward_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto* iptr = (uint8_t*)work_input[0].items();
    int size = work_output[0].n_items * d_itemsize;
    auto* optr = (uint8_t*)work_output[0].items();
    // std::msg_forward(iptr, iptr + size, optr);
    memcpy(optr, iptr, size);

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr