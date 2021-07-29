#include "null_sink_cpu.hh"

namespace gr {
namespace blocks {

null_sink::sptr null_sink::make_cpu(const block_args& args)
{
    return std::make_shared<null_sink_cpu>(args);
}

work_return_code_t null_sink_cpu::work(std::vector<block_work_input>& work_input,
                                         std::vector<block_work_output>& work_output)
{
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr