#include "newblock_cpu.hh"
#include "newblock_cpu_gen.hh"

namespace gr {
namespace newmod {

work_return_code_t newblock_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                  std::vector<block_work_output_sptr>& work_output)
{
    // Do <+signal processing+>
    // Block specific code goes here
    return work_return_code_t::WORK_OK;
}


} // namespace newmod
} // namespace gr