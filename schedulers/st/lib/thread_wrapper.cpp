#include "thread_wrapper.hpp"

#include <gnuradio/thread.hpp>
#include <gnuradio/concurrent_queue.hpp>
#include <gnuradio/flowgraph_monitor.hpp>
#include <gnuradio/scheduler_message.hpp>
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

    canned_notify_all =
        std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_ALL, 0);

    sched_to_notify_upstream.reserve(20);
    sched_to_notify_downstream.reserve(20);

    d_fgmon = fgmon;
    _exec = std::make_unique<graph_executor>(name);
    _exec->initialize(bufman, d_blocks);
    d_thread = std::thread(thread_body, this);
}

void thread_wrapper::start()
{
    // push_message(std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_ALL,
    // 0));
    push_message(canned_notify_all);
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
    // push_message(std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_ALL,
    // 0));
    push_message(canned_notify_all);
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

// std::vector<neighbor_interface_info> thread_wrapper::get_neighbors(nodeid_t blkid)
// {
//     auto ret = get_neighbors_upstream(blkid);
//     auto ds = get_neighbors_downstream(blkid);

//     ret.insert(ret.end(), ds.begin(), ds.end());
//     return ret;
// }

void thread_wrapper::notify_upstream(neighbor_interface_sptr upstream_sched,
                                     nodeid_t blkid)
{
    gr_log_debug(_debug_logger, "notify_upstream");

    upstream_sched->push_message(
        std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_OUTPUT, blkid));
}
void thread_wrapper::notify_downstream(neighbor_interface_sptr downstream_sched,
                                       nodeid_t blkid)
{
    gr_log_debug(_debug_logger, "notify_downstream");
    downstream_sched->push_message(
        std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_INPUT, blkid));
}

void thread_wrapper::handle_parameter_query(std::shared_ptr<param_query_action> item)
{
    auto b = d_block_id_to_block_map[item->block_id()];

    gr_log_debug(
        _debug_logger, "handle parameter query {} - {}", item->block_id(), b->alias());

    b->on_parameter_query(item->param_action());

    if (item->cb_fcn() != nullptr)
        item->cb_fcn()(item->param_action());
}

void thread_wrapper::handle_parameter_change(std::shared_ptr<param_change_action> item)
{
    auto b = d_block_id_to_block_map[item->block_id()];

    gr_log_debug(
        _debug_logger, "handle parameter change {} - {}", item->block_id(), b->alias());

    b->on_parameter_change(item->param_action());

    if (item->cb_fcn() != nullptr)
        item->cb_fcn()(item->param_action());
}


void thread_wrapper::handle_work_notification()
{
    _flags = 0;
    auto s = _exec->run_one_iteration(d_blocks);
    // std::string dbg_work_done;
    // for (auto elem : s) {
    //     dbg_work_done += "[" + std::to_string(elem.first) + "," +
    //                      std::to_string((int)elem.second) + "]" + ",";
    // }
    // gr_log_debug(_debug_logger, dbg_work_done);

    // Based on state of the run_one_iteration, do things
    // If any of the blocks are done, notify the flowgraph monitor
    for (auto& elem : s) {
        if (elem.second == executor_iteration_status::DONE) {
            gr_log_debug(
                _debug_logger, "Signalling DONE to FGM from block {}", elem.first);
            d_fgmon->push_message(
                fg_monitor_message(fg_monitor_message_t::DONE, id(), elem.first));
            break; // only notify the fgmon once
        }
    }

    bool notify_self_ = false;

    sched_to_notify_upstream.clear();
    sched_to_notify_downstream.clear();

    for (auto& elem : s) {

        if (elem.second == executor_iteration_status::READY) {
            // top->notify_neighbors(elem.first);
            neighbor_interface_info info_us, info_ds;
            auto has_us = get_neighbors_upstream(elem.first, info_us);
            auto has_ds = get_neighbors_downstream(elem.first, info_ds);

            if (has_us) {
                sched_to_notify_upstream.push_back(info_us);
                 _flags |= flag_blkd_in;
            }
            if (has_ds) {
                sched_to_notify_downstream.push_back(info_ds);
                _flags |= flag_blkd_out;
            }
            notify_self_ = true;
        } else if (elem.second == executor_iteration_status::BLKD_IN) {
            _flags |= flag_blkd_in;
        } else if (elem.second == executor_iteration_status::BLKD_OUT) {
            _flags |= flag_blkd_out;
        }
    }

    if (notify_self_) {
        gr_log_debug(_debug_logger, "notifying self");
        notify_self();
    }

    if (!sched_to_notify_upstream.empty()) {
        // Reduce to the unique schedulers to notify
        // std::sort(sched_to_notify_upstream.begin(), sched_to_notify_upstream.end());
        // auto last =
        //     std::unique(sched_to_notify_upstream.begin(),
        //     sched_to_notify_upstream.end());
        // sched_to_notify_upstream.erase(last, sched_to_notify_upstream.end());
        for (auto& info : sched_to_notify_upstream) {
            notify_upstream(info.upstream_neighbor_intf, info.upstream_neighbor_blkid);
        }
    }

    if (!sched_to_notify_downstream.empty()) {
        // // Reduce to the unique schedulers to notify
        // std::sort(sched_to_notify_downstream.begin(),
        // sched_to_notify_downstream.end()); auto last =
        // std::unique(sched_to_notify_downstream.begin(),
        //                         sched_to_notify_downstream.end());
        // sched_to_notify_downstream.erase(last, sched_to_notify_downstream.end());
        for (auto& info : sched_to_notify_downstream) {
            int idx = 0;
            for (auto& intf : info.downstream_neighbor_intf) {
                notify_downstream(intf, info.downstream_neighbor_blkids[idx]);
                idx++;
            }
        }
    }
}

void thread_wrapper::thread_body(thread_wrapper* top)
{
    gr_log_info(top->_logger, "starting thread");
    thread::set_thread_name(
        pthread_self(),
        boost::str(boost::format("%s") % top->name())); // % top->id()));

    while (!top->d_thread_stopped) {

        // try to pop messages off the queue
        scheduler_message_sptr msg;
        auto valid = top->pop_message(msg);
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
                    gr_log_debug(top->_debug_logger, "fgm signaled EXIT, exiting thread");
                    // fgmon says that we need to be done, wrap it up
                    // each scheduler could handle this in a different way
                    top->d_thread_stopped = true;
                    break;
                case scheduler_action_t::NOTIFY_OUTPUT:
                    top->_flags &= ~flag_blkd_out;
                    gr_log_debug(
                        top->_debug_logger, "got NOTIFY_OUTPUT from {}", msg->blkid());

                    if (!(top->_flags & flag_blkd_in)) {
                        top->handle_work_notification();
                    }
                    else{
                    gr_log_debug(
                        top->_debug_logger, "Still blocked on the input");
 
                    }
                    break;
                case scheduler_action_t::NOTIFY_INPUT:
                    top->_flags &= ~flag_blkd_in;
                    gr_log_debug(
                        top->_debug_logger, "got NOTIFY_INPUT from {}", msg->blkid());
                    if (!(top->_flags & flag_blkd_out)) {
                        top->handle_work_notification();
                    }
                    else{
                    gr_log_debug(
                        top->_debug_logger, "Still blocked on the output");
 
                    }
                    break;
                case scheduler_action_t::NOTIFY_ALL: {
                    top->_flags = 0x00;
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
        } else {
            std::this_thread::yield();
            gr_log_debug(top->_debug_logger, "No message, do work anyway");
            top->handle_work_notification();
        }
    }
}

} // namespace schedulers
} // namespace gr
