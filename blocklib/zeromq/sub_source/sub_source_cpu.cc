#include "sub_source_cpu.h"
#include "sub_source_cpu_gen.h"

#include <chrono>
#include <thread>

namespace gr {
namespace zeromq {

sub_source_cpu::sub_source_cpu(block_args args)
    : INHERITED_CONSTRUCTORS,
      base_source(ZMQ_SUB,
                  args.itemsize,
                  args.address,
                  args.timeout,
                  args.pass_tags,
                  args.hwm,
                  args.key)
{

    /* Subscribe */
    d_socket.set(zmq::sockopt::subscribe, args.key);
}

work_return_code_t sub_source_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                        std::vector<block_work_output_sptr>& work_output)
{
    auto noutput_items = work_output[0]->n_items;
    bool first = true;
    int done = 0;

    /* Process as much as we can */
    while (1) {
        if (has_pending()) {
            /* Flush anything pending */
            done += flush_pending(
                work_output[0], noutput_items - done, done);

            /* No more space ? */
            if (done == noutput_items)
                break;
        }
        else {
            /* Try to get the next message */
            if (!load_message(first)) {
                // Launch a thread to come back and try again some time later
                std::thread t([this]() {
                    d_debug_logger->debug("ZMQ base_source sleeping");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    this->p_scheduler->push_message(std::make_shared<scheduler_action>(
                        scheduler_action_t::NOTIFY_INPUT));
                });
                t.detach();
                break; /* No message, we're done for now */
            }

            /* Not the first anymore */
            first = false;
        }
    }

    produce_each(done, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace zeromq
} // namespace gr