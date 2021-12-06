#include "nop_cpu.hh"
#include "nop_cpu_gen.hh"

namespace gr {
namespace blocks {

work_return_code_t nop_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                  std::vector<block_work_output_sptr>& work_output)
{
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr