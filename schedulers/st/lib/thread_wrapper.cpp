#include "thread_wrapper.hpp"
#include <thread>

namespace gr {
namespace schedulers {

thread_wrapper::thread_wrapper(const std::string& name,
                               int id,
                               std::vector<block_sptr> blocks,
                               neighbor_interface_map block_sched_map,
                               buffer_manager::sptr bufman,
                               flowgraph_monitor_sptr fgmon)
    : _name(name), _id(id)
{
    _logger = logging::get_logger(name, "default");
    _debug_logger = logging::get_logger(name + "_dbg", "debug");

    d_blocks = blocks;
    d_block_sched_map = block_sched_map;
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
    GR_LOG_DEBUG(_debug_logger, "notify_self");
    push_message(std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_ALL, 0));
}

bool thread_wrapper::get_neighbors_upstream(nodeid_t blkid, neighbor_interface_info& info)
{
    bool ret = false;
    // Find whether this block has an upstream neighbor
    auto search = d_block_sched_map.find(blkid);
    if (search != d_block_sched_map.end()) {
        if (search->second.upstream_neighbor_intf != nullptr) {
            info = search->second;
            return true;
        }
    }

    return ret;
}

bool thread_wrapper::get_neighbors_downstream(nodeid_t blkid,
                                              neighbor_interface_info& info)
{
    // Find whether this block has any downstream neighbors
    auto search = d_block_sched_map.find(blkid);
    if (search != d_block_sched_map.end()) {
        // Entry in the map exists, are there any entries
        if (!search->second.downstream_neighbor_intf.empty()) {
            info = search->second;
            return true;
        }
    }

    return false;
}

void thread_wrapper::notify_upstream(neighbor_interface_sptr upstream_sched,
                                     nodeid_t blkid)
{
    GR_LOG_DEBUG(_debug_logger, "notify_upstream");

    upstream_sched->push_message(
        std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_OUTPUT, blkid));
}
void thread_wrapper::notify_downstream(neighbor_interface_sptr downstream_sched,
                                       nodeid_t blkid)
{
    GR_LOG_DEBUG(_debug_logger, "notify_downstream");
    downstream_sched->push_message(
        std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_INPUT, blkid));
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

    // std::vector<neighbor_interface_info> sched_to_notify_upstream,
    //     sched_to_notify_downstream;

    for (auto elem : s) {

        if (elem.second == executor_iteration_status::READY) {
            //         // top->notify_neighbors(elem.first);
            //         neighbor_interface_info info_us, info_ds;
            //         auto has_us = get_neighbors_upstream(elem.first, info_us);
            //         auto has_ds = get_neighbors_downstream(elem.first, info_ds);

            //         if (has_us) {
            //             sched_to_notify_upstream.push_back(info_us);
            //         }
            //         if (has_ds) {
            //             sched_to_notify_downstream.push_back(info_ds);
            //         }
            notify_self_ = true;
        }
    }

    // if (notify_self_) {
    //     gr_log_debug(_debug_logger, "notifying self");
    //     notify_self();
    // }

    // if (!sched_to_notify_upstream.empty()) {
    //     // Reduce to the unique schedulers to notify
    //     // std::sort(sched_to_notify_upstream.begin(), sched_to_notify_upstream.end());
    //     // auto last =
    //     //     std::unique(sched_to_notify_upstream.begin(),
    //     //     sched_to_notify_upstream.end());
    //     // sched_to_notify_upstream.erase(last, sched_to_notify_upstream.end());
    //     for (auto& info : sched_to_notify_upstream) {
    //         notify_upstream(info.upstream_neighbor_intf, info.upstream_neighbor_blkid);
    //     }
    // }

    // if (!sched_to_notify_downstream.empty()) {
    //     // // Reduce to the unique schedulers to notify
    //     // std::sort(sched_to_notify_downstream.begin(),
    //     // sched_to_notify_downstream.end()); auto last =
    //     // std::unique(sched_to_notify_downstream.begin(),
    //     //                         sched_to_notify_downstream.end());
    //     // sched_to_notify_downstream.erase(last, sched_to_notify_downstream.end());
    //     for (auto& info : sched_to_notify_downstream) {
    //         int idx = 0;
    //         for (auto& intf : info.downstream_neighbor_intf) {
    //             notify_downstream(intf, info.downstream_neighbor_blkids[idx]);
    //             idx++;
    //         }
    //     }
    // }

    return notify_self_;
}


void thread_wrapper::thread_body(thread_wrapper* top)
{
    GR_LOG_INFO(top->_logger, "starting thread");

    bool blocking_queue = true;
    while (!top->d_thread_stopped) {

        scheduler_message_sptr msg;

        // try to pop messages off the queue
        bool valid = true;
        bool do_some_work = false;
        while (valid) {
            if (blocking_queue)
            {
                valid = top->pop_message(msg);
            } else
            {
                valid = top->pop_message_nonblocking(msg);
            }

            blocking_queue = false;
            
            if (valid) // this blocks
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
                case scheduler_message_t::MSGPORT_MESSAGE:
                {
                
                    auto m = std::static_pointer_cast<msgport_message>(msg);
                    m->callback()(m->message());

                    break;
                }
                default:
                    break;
                }
            }
        }

        bool work_returned_ready = false;
        if (do_some_work) {
            work_returned_ready = top->handle_work_notification();
        }

        if (!work_returned_ready)
        {
            blocking_queue = true;
        }
    }
}

} // namespace schedulers
} // namespace gr
