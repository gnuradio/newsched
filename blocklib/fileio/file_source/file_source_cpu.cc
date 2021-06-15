#include "file_source_cpu.hh"

namespace gr {
namespace fileio {

file_source::sptr file_source::make_cpu(const block_args& args) { return std::make_shared<file_source_cpu>(args); }

work_return_code_t file_source_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto* iptr = (uint8_t*)work_input[0].items();
    int size = work_output[0].n_items * d_itemsize;
    auto* optr = (uint8_t*)work_output[0].items();
    // std::file_source(iptr, iptr + size, optr);
    memcpy(optr, iptr, size);

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}


} // namespace fileio
} // namespace gr