//#include <gnuradio/scheduler.hpp>
#include <gnuradio/scheduler.hpp>
// #include <boost/circular_buffer.hpp>
#include <gnuradio/concurrent_queue.hpp>
#include <gnuradio/domain_adapter.hpp>
#include <gnuradio/scheduler_message.hpp>
#include <gnuradio/simplebuffer.hpp>
#include <map>
#include <thread> // std::thread

namespace gr {

namespace schedulers {

class scheduler_st : public scheduler
{
private:
    std::string _name;

public:
    scheduler_sync* sched_sync;
    const int s_fixed_buf_size;
    static const int s_min_items_to_process = 1;
    const size_t s_max_buf_items; // = s_fixed_buf_size / 2;
    const size_t s_min_buf_items = 1;

    typedef std::shared_ptr<scheduler_st> sptr;

    scheduler_st(const std::string name = "single_threaded",
                 const unsigned int fixed_buf_size = 8192);
    ~scheduler_st(){

    };

    int get_buffer_num_items(edge e, flat_graph_sptr fg);
    void initialize(flat_graph_sptr fg,
                    flowgraph_monitor_sptr fgmon,
                    block_scheduler_map block_sched_map);
    void start();
    void stop();
    void wait();
    void run();

    std::map<nodeid_t, scheduler_iteration_status>
    run_one_iteration(std::vector<block_sptr> blocks = std::vector<block_sptr>());

    void notify_self();
    std::vector<scheduler_sptr> get_neighbors_upstream(nodeid_t blkid);

    std::vector<scheduler_sptr> get_neighbors_downstream(nodeid_t blkid);

    std::vector<scheduler_sptr> get_neighbors(nodeid_t blkid);

    void notify_upstream(scheduler_sptr upstream_sched);
    void notify_downstream(scheduler_sptr downstream_sched);
    void handle_parameter_query(std::shared_ptr<param_query_action> item);
    void handle_parameter_change(std::shared_ptr<param_change_action> item);
    void handle_work_notification();

private:
    flat_graph_sptr d_fg;
    block_scheduler_map
        d_block_sched_map; // map of block ids to scheduler interfaces / adapters
    flowgraph_monitor_sptr d_fgmon;
    std::vector<block_sptr> d_blocks;
    std::map<nodeid_t, block_sptr> d_block_id_to_block_map;
    std::map<std::string, edge> d_edge_catalog;
    std::map<std::string, buffer_sptr> d_edge_buffers;
    std::map<port_sptr, std::vector<buffer_sptr>> d_block_buffers;
    std::thread d_thread;
    bool d_thread_stopped = false;


    static void thread_body(scheduler_st* top);
};
} // namespace schedulers
} // namespace gr
