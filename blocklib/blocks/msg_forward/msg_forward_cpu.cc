#include "msg_forward_cpu.hh"

namespace gr {
namespace blocks {

msg_forward::sptr msg_forward::make_cpu(const block_args& args) { return std::make_shared<msg_forward_cpu>(args); }

work_return_code_t msg_forward_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    // there are no work inputs or outputs -- not sure why this would need to get called
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr
