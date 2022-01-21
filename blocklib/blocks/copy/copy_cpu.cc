#include "copy_cpu.hh"
#include "copy_cpu_gen.hh"
namespace gr {
namespace blocks {

copy_cpu::copy_cpu(block_args args) : INHERITED_CONSTRUCTORS {}
work_return_code_t copy_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                  std::vector<block_work_output_sptr>& work_output)
{
    auto iptr = work_input[0]->items<uint8_t>();
    int size = work_output[0]->n_items * work_output[0]->buffer->item_size();
    auto optr = work_output[0]->items<uint8_t>();
    // std::copy(iptr, iptr + size, optr);
    memcpy(optr, iptr, size);

    work_output[0]->n_produced = work_output[0]->n_items;
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr