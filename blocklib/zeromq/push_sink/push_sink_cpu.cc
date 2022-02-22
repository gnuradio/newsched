#include "push_sink_cpu.h"
#include "push_sink_cpu_gen.h"

namespace gr {
namespace zeromq {

push_sink_cpu::push_sink_cpu(block_args args)
    : INHERITED_CONSTRUCTORS,
      base_sink(ZMQ_PUSH,
                     args.itemsize,
                     args.address,
                     args.timeout,
                     args.pass_tags,
                     args.hwm)
{
}
work_return_code_t push_sink_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                       std::vector<block_work_output_sptr>& work_output)
{
    // Poll with a timeout (FIXME: scheduler can't wait for us)
    zmq::pollitem_t itemsout[] = { { static_cast<void*>(d_socket), 0, ZMQ_POLLOUT, 0 } };
    zmq::poll(&itemsout[0], 1, std::chrono::milliseconds{d_timeout});

    // If we can send something, do it
    if (itemsout[0].revents & ZMQ_POLLOUT) {
        work_input[0]->n_consumed = send_message(work_input[0]->raw_items(),
                                                  work_input[0]->n_items,
                                                  work_input[0]->nitems_read());
    }

    return work_return_code_t::WORK_OK;
}


} // namespace zeromq
} // namespace gr