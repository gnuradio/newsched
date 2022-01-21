#include "null_sink_cpu.hh"
#include "null_sink_cpu_gen.hh"

namespace gr {
namespace blocks {

null_sink_cpu::null_sink_cpu(block_args args) : INHERITED_CONSTRUCTORS, d_itemsize(args.itemsize) {}

work_return_code_t null_sink_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                         std::vector<block_work_output_sptr>& work_output)
{
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr