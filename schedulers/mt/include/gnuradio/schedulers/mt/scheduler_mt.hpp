#include <gnuradio/domain.hpp>
#include <gnuradio/domain_adapter_direct.hpp>
#include <gnuradio/graph_utils.hpp>
#include <gnuradio/scheduler.hpp>
#include <gnuradio/schedulers/st/scheduler_st.hpp>

namespace gr {
namespace schedulers {
class scheduler_mt : public scheduler
{
private:
    // std::vector<thread_wrapper::ptr> _threads;
    std::vector<scheduler_sptr> _st_scheds;
    const int s_fixed_buf_size;
    std::map<nodeid_t, scheduler_sptr> _block_thread_map;
    std::vector<std::vector<block_sptr>> _block_groups;

public:
    typedef std::shared_ptr<scheduler_mt> sptr;
    static sptr make(const std::string name = "multi_threaded",
                     const unsigned int fixed_buf_size = 32768)
    {
        return std::make_shared<scheduler_mt>(name, fixed_buf_size);
    }
    scheduler_mt(const std::string name = "multi_threaded",
                 const unsigned int fixed_buf_size = 32768)
        : scheduler(name), s_fixed_buf_size(fixed_buf_size)
    {
        // _default_buf_factory = vmcirc_buffer::make; //simplebuffer::make;
        _default_buf_factory = simplebuffer::make;
    }
    ~scheduler_mt(){};

    void push_message(scheduler_message_sptr msg);
    void add_block_group(std::vector<block_sptr> blocks);

    /**
     * @brief Initialize the multi-threaded scheduler
     *
     * Creates a single-threaded scheduler for each block group, then for each block that
     * is not part of a block group
     *
     * @param fg subgraph assigned to this multi-threaded scheduler
     * @param fgmon sptr to flowgraph monitor object
     * @param block_sched_map for each block in this flowgraph, a map of neighboring
     * schedulers
     */
    void initialize(flat_graph_sptr fg,
                    flowgraph_monitor_sptr fgmon,
                    neighbor_interface_map block_sched_map);
    void start();
    void stop();
    void wait();
    void run();

private:
    /**
     * @brief Append domain adapters to connected block
     *
     * Domain adapters don't show up as blocks, so make sure they get added into the
     * partition configuration
     *
     * @param b block sptr
     * @param fg flowgraph to search edges
     * @param node_vec output node vector that the block and associated domain adapters
     * will be appended to
     */
    void
    append_domain_adapters(block_sptr b, flat_graph_sptr fg, node_vector_t& node_vec);
};
} // namespace schedulers
} // namespace gr