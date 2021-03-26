#pragma once

#include <gnuradio/block_group_properties.hpp>
#include <gnuradio/block.hpp>
#include <gnuradio/concurrent_queue.hpp>
#include <gnuradio/flowgraph_monitor.hpp>
#include <gnuradio/neighbor_interface.hpp>
#include <gnuradio/neighbor_interface_info.hpp>
#include <gnuradio/scheduler_message.hpp>
#include <thread>

#include "graph_executor.hpp"

namespace gr {
namespace schedulers {

/**
 * @brief Wrapper for scheduler thread
 *
 * Creates the worker thread that will process work for all blocks in the graph assigned
 * to this scheduler.  This is the core of the single threaded scheduler.
 *
 */
class thread_wrapper : public neighbor_interface
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

    int _id;
    block_group_properties d_block_group;
    std::vector<block_sptr> d_blocks;

    logger_sptr _logger;
    logger_sptr _debug_logger;

    flowgraph_monitor_sptr d_fgmon;

public:
    typedef std::shared_ptr<thread_wrapper> sptr;

    static sptr make(int id,
                     block_group_properties bgp,
                     buffer_manager::sptr bufman,
                     flowgraph_monitor_sptr fgmon)
    {
        return std::make_shared<thread_wrapper>(id, bgp, bufman, fgmon);
    }

    thread_wrapper(int id,
                   block_group_properties bgp,
                   buffer_manager::sptr bufman,
                   flowgraph_monitor_sptr fgmon);
    int id() { return _id; }
    const std::string& name() { return d_block_group.name(); }

    void push_message(scheduler_message_sptr msg) { msgq.push(msg); }
    bool pop_message(scheduler_message_sptr& msg) { return msgq.pop(msg); }
    bool pop_message_nonblocking(scheduler_message_sptr& msg)
    {
        return msgq.try_pop(msg);
    }

    void start();
    void stop();
    void wait();
    void run();

    bool handle_work_notification();
    static void thread_body(thread_wrapper* top);
};
} // namespace schedulers
} // namespace gr
