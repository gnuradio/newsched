
#include <gnuradio/concurrent_queue.hpp>
#include <gnuradio/flowgraph_monitor.hpp>
#include <gnuradio/scheduler_message.hpp>
#include "graph_executor.hpp"
#include <gnuradio/block.hpp>
#include <gnuradio/neighbor_interface.hpp>
#include <thread>

namespace gr {
namespace schedulers {
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

    void push_message(scheduler_message_sptr msg) { msgq.push(msg); }

    bool pop_message(scheduler_message_sptr& msg) { return msgq.pop(msg); }

    void start();
    void stop();
    void wait();
    void run();

    void notify_self();

    std::vector<neighbor_interface_sptr> get_neighbors_upstream(nodeid_t blkid);

    std::vector<neighbor_interface_sptr> get_neighbors_downstream(nodeid_t blkid);

    std::vector<neighbor_interface_sptr> get_neighbors(nodeid_t blkid);

    void notify_upstream(neighbor_interface_sptr upstream_sched);
    void notify_downstream(neighbor_interface_sptr downstream_sched);
    void handle_parameter_query(std::shared_ptr<param_query_action> item);
    void handle_parameter_change(std::shared_ptr<param_change_action> item);
    void handle_work_notification();
    static void thread_body(thread_wrapper* top);
};
} // namespace schedulers
} // namespace gr