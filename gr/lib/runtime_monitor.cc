#include <gnuradio/logging.hh>
#include <gnuradio/runtime_monitor.hh>
#include <gnuradio/scheduler.hh>
#include <nlohmann/json.hpp>

namespace gr {
runtime_monitor::runtime_monitor(std::vector<std::shared_ptr<scheduler>>& sched_ptrs,
                                     std::vector<std::shared_ptr<runtime_proxy>>& proxy_ptrs,
                                     const std::string& fgname)
    : d_schedulers(sched_ptrs), d_runtime_proxies(proxy_ptrs)
{
    empty_queue();
    _logger = logging::get_logger("runtime_monitor " + fgname, "default");
    _debug_logger = logging::get_logger("runtime_monitor_dbg " + fgname, "debug");
    // Start a monitor thread to keep track of when the schedulers signal info back to
    // the main thread
    GR_LOG_DEBUG(_debug_logger, "Start Runtime Monitor Thread");
    std::thread monitor([this]() {
        while (!_monitor_thread_stopped) {
            // try to pop messages off the queue
            rt_monitor_message_sptr msg;
            if (pop_message(msg)) // this blocks
            {
                if (msg->type() == rt_monitor_message_t::KILL) {
                    GR_LOG_DEBUG(_debug_logger, "KILL");
                    _monitor_thread_stopped = true;
                    for (auto& s : d_runtime_proxies) {
                        s->push_message(
                            rt_monitor_message::make(rt_monitor_message_t::KILL));
                    }
                    break;
                } 
                else if (msg->type() == rt_monitor_message_t::START)
                {
                    GR_LOG_DEBUG(_debug_logger, "START");
                    for (auto s : d_schedulers) {
                        s->start();
                    }
                    for (auto& s : d_runtime_proxies) {
                        if (s->upstream()) {
                        s->push_message(
                            rt_monitor_message::make(rt_monitor_message_t::START));
                        }
                    }
                }
                else if (msg->type() == rt_monitor_message_t::DONE) {
                    GR_LOG_DEBUG(_debug_logger, "DONE");
                    // One scheduler signaled it is done
                    // Notify the other schedulers that they need to flush
                    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // DEBUG
                    for (auto& s : d_schedulers) {
                        GR_LOG_DEBUG(_debug_logger, "a");
                        s->push_message(std::make_shared<scheduler_action>(
                            scheduler_action_t::DONE, 0));
                    }
                    for (auto& s : d_runtime_proxies) {
                        GR_LOG_DEBUG(_debug_logger, "b");
                        s->push_message(
                            rt_monitor_message::make(rt_monitor_message_t::DONE));
                    }
                    GR_LOG_DEBUG(_debug_logger, "c");
                    break;
                }
            }
        }

        GR_LOG_DEBUG(_debug_logger, "Going into second loop");

        std::map<int, bool> sched_done;
        for (auto s : d_schedulers) {
            sched_done[s->id()] = false;
        }
        for (auto s : d_runtime_proxies) {
            sched_done[s->id()] = false;
        }
        while (!_monitor_thread_stopped) {
            // Wait until all the threads are done, then send the EXIT message
            rt_monitor_message_sptr msg;

            if (pop_message(msg)) // this blocks
            {
                if (msg->type() == rt_monitor_message_t::KILL) {
                    GR_LOG_DEBUG(_debug_logger, "KILL2");
                    _monitor_thread_stopped = true;
                    for (auto& s : d_runtime_proxies) {
                        if (s->upstream()) {
                            s->push_message(
                                rt_monitor_message::make(rt_monitor_message_t::KILL));
                        }
                    }
                    break;
                } else if (msg->type() == rt_monitor_message_t::FLUSHED) {
                    sched_done[msg->schedid()] = true;
                    GR_LOG_DEBUG(_debug_logger, "FLUSHED from {}", msg->schedid());

                    bool all_done = true;
                    for (auto s : d_schedulers) {
                        if (!sched_done[s->id()]) {
                            all_done = false;
                        }
                    }
                    // Only check downstream proxies
                    for (auto s : d_runtime_proxies) {
                        if (s->upstream())
                            if (!sched_done[s->id()]) {
                                {
                                    all_done = false;
                                }
                            }
                    }

                    if (all_done) {
                        // Tell the upstream proxies that we are FLUSHED
                        for (auto s : d_runtime_proxies) {
                            if (!s->upstream()) { // If this is downstream, tell the
                                                  // upstream
                                s->push_message(rt_monitor_message::make(
                                    rt_monitor_message_t::FLUSHED));
                            }
                        }
                        for (auto s : d_schedulers) {
                            GR_LOG_DEBUG(_debug_logger, "Telling Schedulers to Exit()");
                            s->push_message(std::make_shared<scheduler_action>(
                                scheduler_action_t::EXIT, 0));
                        }
                        for (auto s : d_runtime_proxies) {
                            if (s->upstream()) {
                                GR_LOG_DEBUG(_debug_logger,
                                             "Telling Downstream Proxies to Exit()");
                                s->push_message(
                                    rt_monitor_message::make(rt_monitor_message_t::KILL));
                            }
                            s->kill();
                        }

                        _monitor_thread_stopped = true;
                    }
                }
            }
        }

        GR_LOG_DEBUG(_debug_logger, "Out of Monitor Thread");
    });
    monitor.detach();
} // TODO: bound the queue size



std::string rt_monitor_message::to_string()
{
    nlohmann::json ret = {
        { "type", string_map[type()] },
        { "schedid", schedid() },
        { "blkid", blkid() },
    };
    return ret.dump();
}

rt_monitor_message::sptr rt_monitor_message::from_string(const std::string& str)
{
    auto json_obj = nlohmann::json::parse(str);

    if (json_obj.count("schedid") && json_obj.count("blkid"))
    {
        return make(rev_string_map[json_obj["type"]], json_obj["schedid"], json_obj["blkid"]);
    }
    else if (json_obj.count("schedid"))
    {
        return make(rev_string_map[json_obj["type"]], json_obj["schedid"]);
    }
    else
    {
        return make(rev_string_map[json_obj["type"]]);
    }
}
    

std::map<rt_monitor_message_t, std::string> rt_monitor_message::string_map = {
    { rt_monitor_message_t::UNKNOWN, "UNKNOWN" },
    { rt_monitor_message_t::DONE, "DONE" },
    { rt_monitor_message_t::FLUSHED, "FLUSHED" },
    { rt_monitor_message_t::KILL, "KILL" },
    { rt_monitor_message_t::START, "START" }
};

std::map<std::string, rt_monitor_message_t> rt_monitor_message::rev_string_map = {
    { "UNKNOWN", rt_monitor_message_t::UNKNOWN },
    { "DONE", rt_monitor_message_t::DONE },
    { "FLUSHED", rt_monitor_message_t::FLUSHED },
    { "KILL", rt_monitor_message_t::KILL },
    { "START", rt_monitor_message_t::START }
};

} // namespace gr
