#include "newblock_cpu.hh"

namespace gr {
namespace newmod {

newblock::sptr newblock::make_cpu(const block_args& args) { return std::make_shared<newblock_cpu>(args); }

work_return_code_t newblock_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    // Do <+signal processing+>
    // Block specific code goes here
    return work_return_code_t::WORK_OK;
}


} // namespace newmod
} // namespace gr