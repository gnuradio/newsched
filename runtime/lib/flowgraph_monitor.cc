#include <gnuradio/fgm_proxy.hh>
#include <gnuradio/flowgraph_monitor.hh>
#include <gnuradio/logging.hh>
#include <gnuradio/scheduler.hh>

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
            fg_monitor_message_sptr msg;
            if (pop_message(msg)) // this blocks
            {
                if (msg->type() == fg_monitor_message_t::KILL) {
                    GR_LOG_DEBUG(_debug_logger, "KILL");
                    _monitor_thread_stopped = true;
                    for (auto& s : d_fgm_proxies) {
                        s->push_message(
                            fg_monitor_message::make(fg_monitor_message_t::KILL));
                    }
                    break;
                } else if (msg->type() == fg_monitor_message_t::DONE) {
                    GR_LOG_DEBUG(_debug_logger, "DONE");
                    // One scheduler signaled it is done
                    // Notify the other schedulers that they need to flush
                    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // DEBUG
                    for (auto& s : d_schedulers) {
                        s->push_message(std::make_shared<scheduler_action>(
                            scheduler_action_t::DONE, 0));
                    }
                    for (auto& s : d_fgm_proxies) {
                        s->push_message(
                            fg_monitor_message::make(fg_monitor_message_t::DONE));
                    }
                    break;
                }
            }
        }

        std::map<int, bool> sched_done;
        for (auto s : d_schedulers) {
            sched_done[s->id()] = false;
        }
        for (auto s : d_fgm_proxies) {
            sched_done[s->id()] = false;
        }
        while (!_monitor_thread_stopped) {
            // Wait until all the threads are done, then send the EXIT message
            fg_monitor_message_sptr msg;

            if (pop_message(msg)) // this blocks
            {
                if (msg->type() == fg_monitor_message_t::KILL) {
                    GR_LOG_DEBUG(_debug_logger, "KILL2");
                    _monitor_thread_stopped = true;
                    for (auto& s : d_fgm_proxies) {
                        if (s->upstream()) {
                            s->push_message(
                                fg_monitor_message::make(fg_monitor_message_t::KILL));
                        }
                    }
                    break;
                } else if (msg->type() == fg_monitor_message_t::FLUSHED) {
                    sched_done[msg->schedid()] = true;
                    GR_LOG_DEBUG(_debug_logger, "FLUSHED from {}", msg->schedid());

                    bool all_done = true;
                    for (auto s : d_schedulers) {
                        if (!sched_done[s->id()]) {
                            all_done = false;
                        }
                    }
                    // Only check downstream proxies
                    for (auto s : d_fgm_proxies) {
                        if (s->upstream())
                            if (!sched_done[s->id()]) {
                                {
                                    all_done = false;
                                }
                            }
                    }

                    if (all_done) {
                        // Tell the upstream proxies that we are FLUSHED
                        for (auto s : d_fgm_proxies) {
                            if (!s->upstream()) { // If this is downstream, tell the
                                                  // upstream
                                s->push_message(fg_monitor_message::make(
                                    fg_monitor_message_t::FLUSHED));
                            }
                        }
                        for (auto s : d_schedulers) {
                            GR_LOG_DEBUG(_debug_logger, "Telling Schedulers to Exit()");
                            s->push_message(std::make_shared<scheduler_action>(
                                scheduler_action_t::EXIT, 0));
                        }
                        for (auto s : d_fgm_proxies) {
                            if (s->upstream()) {
                                GR_LOG_DEBUG(_debug_logger,
                                             "Telling Downstream Proxies to Exit()");
                                s->push_message(
                                    fg_monitor_message::make(fg_monitor_message_t::KILL));
                            }
                            s->kill();
                        }

                        _monitor_thread_stopped = true;

                    }
                }
            }
        }
    });
    monitor.detach();
} // namespace gr


std::map<fg_monitor_message_t, std::string> fg_monitor_message::string_map = {
    { fg_monitor_message_t::UNKNOWN, "UNKNOWN" },
    { fg_monitor_message_t::DONE, "DONE" },
    { fg_monitor_message_t::FLUSHED, "FLUSHED" },
    { fg_monitor_message_t::KILL, "KILL" },
    { fg_monitor_message_t::START, "START" }
};

std::map<std::string, fg_monitor_message_t> fg_monitor_message::rev_string_map = {
    { "UNKNOWN", fg_monitor_message_t::UNKNOWN },
    { "DONE", fg_monitor_message_t::DONE },
    { "FLUSHED", fg_monitor_message_t::FLUSHED },
    { "KILL", fg_monitor_message_t::KILL },
    { "START", fg_monitor_message_t::START }
};


} // namespace gr
