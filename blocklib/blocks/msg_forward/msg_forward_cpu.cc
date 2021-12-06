#include "msg_forward_cpu.hh"
#include "msg_forward_cpu_gen.hh"

namespace gr {
namespace blocks {

work_return_code_t msg_forward_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                  std::vector<block_work_output_sptr>& work_output)
{
    // there are no work inputs or outputs -- not sure why this would need to get called
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr
