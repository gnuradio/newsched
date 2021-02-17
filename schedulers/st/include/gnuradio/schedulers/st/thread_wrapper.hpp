#pragma once

#include "graph_executor.hpp"
#include <gnuradio/block.hpp>
#include <gnuradio/concurrent_queue.hpp>
#include <gnuradio/flowgraph_monitor.hpp>
#include <gnuradio/neighbor_interface_info.hpp>
#include <gnuradio/scheduler_message.hpp>
#include <thread>

namespace gr {
namespace schedulers {

/**
 * @brief Wrapper for scheduler thread
 *
 * Creates the worker thread that will process work for all blocks in the graph assigned
 * to this scheduler.  This is the core of the single threaded scheduler.
 *
 */
class thread_wrapper
{
private:
    /**
     * @brief Single message queue for all types of messages to this thread
     *
     */
    concurrent_queue<scheduler_message_sptr> msgq;
    std::thread d_thread;
    bool d_thread_stopped = false;
    std::unique_ptr<graph_executor> _exec;

    std::vector<block_sptr> d_blocks;

    logger_sptr _logger;
    logger_sptr _debug_logger;
    neighbor_interface_map
        d_block_sched_map; // map of block ids to scheduler interfaces / adapters
    std::map<nodeid_t, block_sptr> d_block_id_to_block_map;

    flowgraph_monitor_sptr d_fgmon;
    std::string _name;
    int _id;

public:
    typedef std::unique_ptr<thread_wrapper> ptr;

    static ptr make(const std::string& name,
                    int id,
                    std::vector<block_sptr> blocks,
                    neighbor_interface_map block_sched_map,
                    buffer_manager::sptr bufman,
                    flowgraph_monitor_sptr fgmon)
    {
        return std::make_unique<thread_wrapper>(
            name, id, blocks, block_sched_map, bufman, fgmon);
    }

    thread_wrapper(const std::string& name,
                   int id,
                   std::vector<block_sptr> blocks,
                   neighbor_interface_map block_sched_map,
                   buffer_manager::sptr bufman,
                   flowgraph_monitor_sptr fgmon);
    int id() { return _id; }
    void set_id(int id) { _id = id; }
    const std::string& name() { return _name; }
    void set_name(int name) { _name = name; }

    void push_message(scheduler_message_sptr msg) { msgq.push(msg); }
    bool pop_message(scheduler_message_sptr& msg) { return msgq.pop(msg); }
    bool pop_message_nonblocking(scheduler_message_sptr& msg) { return msgq.try_pop(msg); }

    void start();
    void stop();
    void wait();
    void run();

    void notify_self();

    bool get_neighbors_upstream(nodeid_t blkid, neighbor_interface_info& info);
    bool get_neighbors_downstream(nodeid_t blkid, neighbor_interface_info& info);

    void notify_upstream(neighbor_interface_sptr upstream_sched, nodeid_t blkid);
    void notify_downstream(neighbor_interface_sptr downstream_sched, nodeid_t blkid);
    bool handle_work_notification();
    static void thread_body(thread_wrapper* top);
};
} // namespace schedulers
} // namespace gr
