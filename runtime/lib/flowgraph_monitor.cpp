#include <gnuradio/flowgraph_monitor.hpp>

#include <gnuradio/scheduler.hpp>

#include <gnuradio/logging.hpp>

namespace gr {
void flowgraph_monitor::start()
{
    empty_queue();
    // Start a monitor thread to keep track of when the schedulers signal info back to
    // the main thread
    std::thread monitor([this]() {
        auto _logger = logging::get_logger("flowgraph_monitor", "default");
        auto _debug_logger = logging::get_logger("flowgraph_monitor_dbg", "debug");
        while (!_monitor_thread_stopped) {
            // try to pop messages off the queue
            fg_monitor_message msg;
            if (pop_message(msg)) // this blocks
            {
                if (msg.type() == fg_monitor_message_t::KILL) {
                    gr_log_debug(_debug_logger, "KILL");
                    _monitor_thread_stopped = true;
                    break;
                } else if (msg.type() == fg_monitor_message_t::DONE) {
                    gr_log_debug(_debug_logger, "DONE");
                    // One scheduler signaled it is done
                    // Notify the other schedulers that they need to flush
                    for (auto& s : d_schedulers) {
                        s->push_message(
                            std::make_shared<scheduler_action>(scheduler_action_t::DONE));
                    }
                    break;
                }
            }
        }


        std::map<int, bool> sched_done;
        for (auto s : d_schedulers) {
            sched_done[s->id()] = false;
        }
        while (!_monitor_thread_stopped) {
            // Wait until all the threads are done, then send the EXIT message
            fg_monitor_message msg;

            if (pop_message(msg)) // this blocks
            {
                if (msg.type() == fg_monitor_message_t::KILL) {
                    gr_log_debug(_debug_logger, "KILL2");
                    _monitor_thread_stopped = true;
                    break;
                } else if (msg.type() == fg_monitor_message_t::FLUSHED) {
                    sched_done[msg.schedid()] = true;
                    gr_log_debug(_debug_logger, "FLUSHED");

                    bool all_done = true;
                    for (auto s : d_schedulers) {
                        if (!sched_done[s->id()]) {
                            all_done = false;
                        }
                    }

                    if (all_done) {
                        for (auto s : d_schedulers) {
                            gr_log_debug(_debug_logger, "Telling Schedulers to Exit()");
                            s->push_message(std::make_shared<scheduler_action>(
                                scheduler_action_t::EXIT));
                        }
                    }
                }
            }
        }
    });
    monitor.detach();
}

} // namespace gr