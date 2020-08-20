//#include <gnuradio/scheduler.hpp>
#include <gnuradio/scheduler.hpp>
// #include <boost/circular_buffer.hpp>
#include <gnuradio/domain_adapter.hpp>
#include <gnuradio/simplebuffer.hpp>
#include <thread> // std::thread
namespace gr {

namespace schedulers {

// This is a terrible scheduler just to pass data through the blocks
// and demonstrate the pluggable scheduler concept

class scheduler_simplestream : public scheduler
{
private:
    std::string _name;

public:
    scheduler_sync* sched_sync;
    const int s_fixed_buf_size;
    static const int s_min_items_to_process = 1;
    const int s_max_buf_items; // = s_fixed_buf_size / 2;

    typedef std::shared_ptr<scheduler_simplestream> sptr;

    scheduler_simplestream(const std::string name = "simplestream",
                           const unsigned int fixed_buf_size = 8192)
        : scheduler(name),
          s_fixed_buf_size(fixed_buf_size),
          s_max_buf_items(fixed_buf_size / 2)
    {
    }
    ~scheduler_simplestream(){

    };

    void initialize(flat_graph_sptr fg);

    void start(scheduler_sync* sync);
    void stop();

    void wait();

    void run(scheduler_sync* sync);

private:
    flat_graph_sptr d_fg;
    std::vector<block_sptr> d_blocks;
    std::map<std::string, edge> d_edge_catalog;
    std::map<std::string, buffer_sptr> d_edge_buffers;
    // std::map<block_sptr, std::map<block::io, std::map<int, simplebuffer::sptr>>>
    //     d_block_buffers; // store the block buffers
    std::map<port_sptr, std::vector<buffer_sptr>> d_block_buffers;
    // std::map<port_sptr, simplebuffer::sptr> d_block_buffers; // store the block buffers
    std::thread d_thread;
    bool d_thread_stopped = false;

    static void thread_body(scheduler_simplestream* top);
};
} // namespace schedulers
} // namespace gr
