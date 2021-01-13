//#include <gnuradio/scheduler.hpp>
#include <gnuradio/scheduler.hpp>
// #include <boost/circular_buffer.hpp>
#include "graph_executor.hpp"
#include "thread_wrapper.hpp"
#include <gnuradio/concurrent_queue.hpp>
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
    thread_wrapper::ptr _thread;

    bool pop_message(scheduler_message_sptr& msg) { return _thread->pop_message(msg); }
    
public:
    const int s_fixed_buf_size;

    typedef std::shared_ptr<scheduler_st> sptr;

    static sptr make(const std::string name = "single_threaded",
                      const unsigned int fixed_buf_size = 32768)
    {
        return std::make_shared<scheduler_st>(name, fixed_buf_size);
    }
    scheduler_st(const std::string name = "single_threaded",
                 const unsigned int fixed_buf_size = 32768);
    ~scheduler_st(){};

    void push_message(scheduler_message_sptr msg) { _thread->push_message(msg); }


    void initialize(flat_graph_sptr fg,
                    flowgraph_monitor_sptr fgmon);
    void start();
    void stop();
    void wait();
    void run();


private:
    flat_graph_sptr d_fg;
};
} // namespace schedulers
} // namespace gr
