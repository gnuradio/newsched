#include "pub_sink_cpu.hh"
#include "pub_sink_cpu_gen.hh"

namespace gr {
namespace zeromq {

pub_sink_cpu::pub_sink_cpu(block_args args)
    : INHERITED_CONSTRUCTORS,
      base_sink(
          ZMQ_PUB, args.itemsize, args.address, args.timeout, args.pass_tags, args.hwm, args.key)
{
}

work_return_code_t pub_sink_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                      std::vector<block_work_output_sptr>& work_output)
{
    auto noutput_items = work_input[0]->n_items;
    auto nread = work_input[0]->nitems_read();
    auto nsent = send_message(work_input[0]->raw_items(), noutput_items, nread);

    consume_each(nsent, work_input);
    return work_return_code_t::WORK_OK;
}


} // namespace zeromq
} // namespace gr