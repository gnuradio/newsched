#include <gnuradio/block_group_properties.hpp>
#include <gnuradio/domain.hpp>
#include <gnuradio/graph_utils.hpp>
#include <gnuradio/scheduler.hpp>
#include <gnuradio/vmcircbuf.hpp>

#include "thread_wrapper.hpp"
namespace gr {
namespace schedulers {
class scheduler_mt : public scheduler
{
private:
    std::vector<thread_wrapper::sptr> _threads;
    const int s_fixed_buf_size;
    std::map<nodeid_t, neighbor_interface_sptr> _block_thread_map;
    std::vector<block_group_properties> _block_groups;

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
        _default_buf_properties =
            vmcirc_buffer_properties::make(vmcirc_buffer_type::AUTO);
    }
    ~scheduler_mt(){};

    void push_message(scheduler_message_sptr msg);
    void add_block_group(const std::vector<block_sptr>& blocks,
                         const std::string& name = "",
                         const std::vector<unsigned int>& affinity_mask = {});

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
    void initialize(flat_graph_sptr fg, flowgraph_monitor_sptr fgmon);
    void start();
    void stop();
    void wait();
    void run();
};
} // namespace schedulers
} // namespace gr