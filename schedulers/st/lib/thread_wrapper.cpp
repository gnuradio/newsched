#include "thread_wrapper.hpp"
#include <thread>

namespace gr {
namespace schedulers {

thread_wrapper::thread_wrapper(const std::string& name,
                               int id,
                               std::vector<block_sptr> blocks,
                               buffer_manager::sptr bufman,
                               flowgraph_monitor_sptr fgmon)
    : _name(name), _id(id)
{
    _logger = logging::get_logger(name, "default");
    _debug_logger = logging::get_logger(name + "_dbg", "debug");

    d_blocks = blocks;
    for (auto b : d_blocks) {
        d_block_id_to_block_map[b->id()] = b;
    }

    d_fgmon = fgmon;
    _exec = std::make_unique<graph_executor>(name);
    _exec->initialize(bufman, d_blocks);
    d_thread = std::thread(thread_body, this);
}

void thread_wrapper::start()
{
    push_message(std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_ALL, 0));
}
void thread_wrapper::stop()
{
    d_thread_stopped = true;
    push_message(std::make_shared<scheduler_action>(scheduler_action_t::EXIT, 0));
    d_thread.join();
    for (auto& b : d_blocks) {
        b->stop();
    }
}
void thread_wrapper::wait()
{
    d_thread.join();
    for (auto& b : d_blocks) {
        b->done();
    }
}
void thread_wrapper::run()
{
    start();
    wait();
}

void thread_wrapper::notify_self()
{
    gr_log_debug(_debug_logger, "notify_self");
    push_message(std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_ALL, 0));
}

void thread_wrapper::handle_work_notification()
{
    auto s = _exec->run_one_iteration(d_blocks);

    // Based on state of the run_one_iteration, do things
    // If any of the blocks are done, notify the flowgraph monitor
    for (auto elem : s) {
        if (elem.second == executor_iteration_status::DONE) {
            gr_log_debug(
                _debug_logger, "Signalling DONE to FGM from block {}", elem.first);
            d_fgmon->push_message(
                fg_monitor_message(fg_monitor_message_t::DONE, id(), elem.first));
            break; // only notify the fgmon once
        }
    }

    bool notify_self_ = false;
    for (auto elem : s) {

        if (elem.second == executor_iteration_status::READY) {
            notify_self_ = true;
        }
    }

    if (notify_self_) {
        gr_log_debug(_debug_logger, "notifying self");
        notify_self();
    }
}

void thread_wrapper::thread_body(thread_wrapper* top)
{
    gr_log_info(top->_logger, "starting thread");
    while (!top->d_thread_stopped) {

        // try to pop messages off the queue
        scheduler_message_sptr msg;
        if (top->pop_message(msg)) // this blocks
        {
            switch (msg->type()) {
            case scheduler_message_t::SCHEDULER_ACTION: {
                // Notification that work needs to be done
                // either from runtime or upstream or downstream or from self

                auto action = std::static_pointer_cast<scheduler_action>(msg);
                switch (action->action()) {
                case scheduler_action_t::DONE:
                    // fgmon says that we need to be done, wrap it up
                    // each scheduler could handle this in a different way
                    gr_log_debug(top->_debug_logger,
                                 "fgm signaled DONE, pushing flushed");
                    top->d_fgmon->push_message(
                        fg_monitor_message(fg_monitor_message_t::FLUSHED, top->id()));
                    break;
                case scheduler_action_t::EXIT:
                    gr_log_debug(top->_debug_logger, "fgm signaled EXIT, exiting thread");
                    // fgmon says that we need to be done, wrap it up
                    // each scheduler could handle this in a different way
                    top->d_thread_stopped = true;
                    break;
                case scheduler_action_t::NOTIFY_OUTPUT:
                    gr_log_debug(
                        top->_debug_logger, "got NOTIFY_OUTPUT from {}", msg->blkid());
                    top->handle_work_notification();
                    break;
                case scheduler_action_t::NOTIFY_INPUT:
                    gr_log_debug(
                        top->_debug_logger, "got NOTIFY_INPUT from {}", msg->blkid());
                    top->handle_work_notification();
                    break;
                case scheduler_action_t::NOTIFY_ALL: {
                    gr_log_debug(
                        top->_debug_logger, "got NOTIFY_ALL from {}", msg->blkid());
                    top->handle_work_notification();
                    break;
                }
                default:
                    break;
                    break;
                }
                break;
            }
            default:
                break;
            }
        }
    }
}

} // namespace schedulers
} // namespace gr