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
                    GR_LOG_DEBUG(_debug_logger, "KILL");
                    _monitor_thread_stopped = true;
                    break;
                } else if (msg.type() == fg_monitor_message_t::DONE) {
                    GR_LOG_DEBUG(_debug_logger, "DONE");
                    // One scheduler signaled it is done
                    // Notify the other schedulers that they need to flush
                    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // DEBUG
                    for (auto& s : d_schedulers) {
                        s->push_message(
                            std::make_shared<scheduler_action>(scheduler_action_t::DONE, 0));
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
                    GR_LOG_DEBUG(_debug_logger, "KILL2");
                    _monitor_thread_stopped = true;
                    break;
                } else if (msg.type() == fg_monitor_message_t::FLUSHED) {
                    sched_done[msg.schedid()] = true;
                    GR_LOG_DEBUG(_debug_logger, "FLUSHED");

                    bool all_done = true;
                    for (auto s : d_schedulers) {
                        if (!sched_done[s->id()]) {
                            all_done = false;
                        }
                    }

                    if (all_done) {
                        for (auto s : d_schedulers) {
                            GR_LOG_DEBUG(_debug_logger, "Telling Schedulers to Exit()");
                            s->push_message(std::make_shared<scheduler_action>(
                                scheduler_action_t::EXIT, 0));
                        }
                    }
                }
            }
        }
    });
    monitor.detach();
}

bool flowgraph_monitor::replace_scheduler(
    std::shared_ptr<scheduler> original,
    const std::vector<std::shared_ptr<scheduler>> replacements)
{
    // find original in d_schedulers
    auto it = std::find(d_schedulers.begin(), d_schedulers.end(), original);
    if (it != d_schedulers.end()) {
        d_schedulers.erase(it);
        // replace it with the specified replacements
        d_schedulers.insert( d_schedulers.end(), replacements.begin(), replacements.end() );
        return true;
    } else {
        return false;
    }
    
}

} // namespace gr
