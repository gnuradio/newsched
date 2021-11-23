#include "thread_wrapper.hh"
#include <gnuradio/thread.hh>
#include <boost/format.hpp>
#include <thread>

namespace gr {
namespace schedulers {

thread_wrapper::thread_wrapper(int id,
                               block_group_properties bgp,
                               buffer_manager::sptr bufman,
                               flowgraph_monitor_sptr fgmon)
    : _id(id), d_block_group(bgp), d_blocks(bgp.blocks())
{
    _logger = logging::get_logger(bgp.name(), "default");
    _debug_logger = logging::get_logger(bgp.name() + "_dbg", "debug");

    for (auto b : d_blocks) {
        d_block_id_to_block_map[b->id()] = b;
    }

    d_fgmon = fgmon;
    _exec = std::make_unique<graph_executor>(bgp.name());
    _exec->initialize(bufman, d_blocks);
    d_thread = std::thread(thread_body, this);
}

void thread_wrapper::start()
{
    for (auto& b : d_blocks) {
        b->start();
    }
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

bool thread_wrapper::handle_work_notification()
{
    auto s = _exec->run_one_iteration(d_blocks);

    // Based on state of the run_one_iteration, do things
    // If any of the blocks are done, notify the flowgraph monitor
    for (auto elem : s) {
        if (elem.second == executor_iteration_status::DONE) {
            GR_LOG_DEBUG(
                _debug_logger, "Signalling DONE to FGM from block {}", elem.first);
            d_fgmon->push_message(
                fg_monitor_message(fg_monitor_message_t::DONE, id(), elem.first));
            break; // only notify the fgmon once
        }
    }

    bool notify_self_ = false;
    bool kick = false;
    bool all_blkd = true;
    for (auto elem : s) {
        if (elem.second == executor_iteration_status::READY ||
            elem.second == executor_iteration_status::BLKD_OUT) {
            notify_self_ = true;
        } else if (elem.second == executor_iteration_status::BLKD_IN) {
            kick = true;
        }

        if (elem.second != executor_iteration_status::BLKD_IN &&
            elem.second != executor_iteration_status::BLKD_OUT) {
            // Ignore source blocks
            if (!d_block_id_to_block_map[elem.first]->input_stream_ports().size())
            {
                all_blkd = false;
            }
        }
    }

    if (d_flushing) { 
        if (all_blkd) {
            if (++d_flush_cnt >= 3) {
                gr_log_debug(_debug_logger, "All blocks in thread {} blocked, pushing flushed", id());
                d_fgmon->push_message(
                    fg_monitor_message(fg_monitor_message_t::FLUSHED, id()));
                return false;
            }
        }
        else
        {
            d_flush_cnt = 0;
            gr_log_debug(_debug_logger, "Not all blocks reporting BLKD");
        }
    }


    if (kick) {
        // after some period of time, drop a message in the queue to try again on this
        // thread
        std::thread th([this] {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(100)); // make configurable
            this->push_message(
                std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_INPUT, 0));
        });

        th.detach();
    }

    return notify_self_;
}

void thread_wrapper::handle_parameter_query(std::shared_ptr<param_query_action> item)
{
    auto b = d_block_id_to_block_map[item->blkid()];

    gr_log_debug(
        _debug_logger, "handle parameter query {} - {}", item->blkid(), b->alias());

    b->on_parameter_query(item->param_action());

    if (item->cb_fcn() != nullptr)
        item->cb_fcn()(item->param_action());
}

void thread_wrapper::handle_parameter_change(std::shared_ptr<param_change_action> item)
{
    auto b = d_block_id_to_block_map[item->blkid()];

    gr_log_debug(
        _debug_logger, "handle parameter change {} - {}", item->blkid(), b->alias());

    b->on_parameter_change(item->param_action());

    if (item->cb_fcn() != nullptr)
        item->cb_fcn()(item->param_action());
}


void thread_wrapper::thread_body(thread_wrapper* top)
{
    GR_LOG_INFO(top->_logger, "starting thread");

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <windows.h>
    thread::set_thread_name(
        GetCurrentThread(),
        boost::str(boost::format("%s%d") % block->name() % block->unique_id()));
#else
    thread::set_thread_name(pthread_self(),
                            str(boost::format("%s%d") % top->d_block_group.name() %
                                top->d_block_group.blocks()[0]->id()));
#endif

    // Set thread affinity if it was set before fg was started.
    if (!top->d_block_group.processor_affinity().empty()) {
        std::cout << "setting affinity of thread " << thread::get_current_thread_id()
                  << " to " << top->d_block_group.processor_affinity()[0] << std::endl;
        gr::thread::thread_bind_to_processor(thread::get_current_thread_id(),
                                             top->d_block_group.processor_affinity());
    }

    // // Set thread priority if it was set before fg was started
    // if (block->thread_priority() > 0) {
    //     gr::thread::set_thread_priority(d->thread, block->thread_priority());
    // }

    bool blocking_queue = true;
    while (!top->d_thread_stopped) {

        scheduler_message_sptr msg;

        // try to pop messages off the queue
        bool valid = true;
        bool do_some_work = false;
        while (valid) {
            if (blocking_queue) {
                valid = top->pop_message(msg);
            } else {
                valid = top->pop_message_nonblocking(msg);
            }

            blocking_queue = false;

            if (valid) {
                switch (msg->type()) {
                case scheduler_message_t::SCHEDULER_ACTION: {
                    // Notification that work needs to be done
                    // either from runtime or upstream or downstream or from self

                    auto action = std::static_pointer_cast<scheduler_action>(msg);
                    switch (action->action()) {
                    case scheduler_action_t::DONE:
                        // fgmon says that we need to be done, wrap it up
                        // each scheduler could handle this in a different way

                        // if a block (e.g. head block) has told the fgmon that we are
                        // done, need to push the rest of the samples through the pipeline
                        // -- hang in this state until all the blocks in this thread
                        // report
                        //    either BLKD_IN or BLKD_OUT

                        top->start_flushing();
                        do_some_work = true;

                        break;
                    case scheduler_action_t::EXIT:
                        gr_log_debug(top->_debug_logger,
                                     "fgm signaled EXIT, exiting thread");
                        // fgmon says that we need to be done, wrap it up
                        // each scheduler could handle this in a different way
                        top->d_thread_stopped = true;
                        break;
                    case scheduler_action_t::NOTIFY_OUTPUT:
                        gr_log_debug(top->_debug_logger,
                                     "got NOTIFY_OUTPUT from {}",
                                     msg->blkid());
                        do_some_work = true;
                        break;
                    case scheduler_action_t::NOTIFY_INPUT:
                        gr_log_debug(
                            top->_debug_logger, "got NOTIFY_INPUT from {}", msg->blkid());

                        do_some_work = true;
                        break;
                    case scheduler_action_t::NOTIFY_ALL: {
                        gr_log_debug(
                            top->_debug_logger, "got NOTIFY_ALL from {}", msg->blkid());
                        do_some_work = true;
                        break;
                    }
                    default:
                        break;
                    }
                    break;
                }
                case scheduler_message_t::MSGPORT_MESSAGE: {

                    auto m = std::static_pointer_cast<msgport_message>(msg);
                    m->callback()(m->message());

                    break;
                }
                case scheduler_message_t::PARAMETER_QUERY: {
                    // Query the state of a parameter on a block
                    top->handle_parameter_query(
                        std::static_pointer_cast<param_query_action>(msg));
                } break;
                case scheduler_message_t::PARAMETER_CHANGE: {
                    // Query the state of a parameter on a block
                    top->handle_parameter_change(
                        std::static_pointer_cast<param_change_action>(msg));
                } break;
                default:
                    break;
                }
            }
        }

        bool work_returned_ready = false;
        if (do_some_work) {
            work_returned_ready = top->handle_work_notification();
        }

        if (!work_returned_ready) {
            blocking_queue = true;
        }
    }
}

} // namespace schedulers
} // namespace gr
