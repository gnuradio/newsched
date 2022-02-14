#pragma once

#include <gnuradio/flowgraph.hh>
#include <gnuradio/scheduler.hh>
#include <gnuradio/runtime_monitor.hh>
#include <gnuradio/runtime_proxy.hh>

namespace gr {
class runtime
{
public:
    typedef std::shared_ptr<runtime> sptr; 
    static sptr make() {return std::make_shared<runtime>();}
    runtime();
    void initialize(flowgraph_sptr fg);
    void start();
    void stop();
    void wait();
    void run();
    /**
     * @brief Add a scheduler via a pair of scheduler and vector of blocks
     * 
     * @param conf 
     */
    void add_scheduler(std::pair<scheduler_sptr, std::vector<node_sptr>> conf);
    /**
     * @brief Add a scheduler with no associated blocks
     * 
     * This Scheduler will be assigned all the blocks in the flowgraph
     * 
     * @param sched 
     */
    void add_scheduler(scheduler_sptr sched);

    void add_proxy(runtime_proxy_sptr proxy);


private:
    bool d_initialized = false;
    std::vector<scheduler_sptr> d_schedulers;
    std::vector<std::vector<node_sptr>> d_blocks_per_scheduler;
    std::vector<std::pair<scheduler_sptr, std::vector<node_sptr>>> d_scheduler_confs;
    const std::string s_default_scheduler_name = "nbt";
    scheduler_sptr d_default_scheduler = nullptr;
    bool d_default_scheduler_inuse = true;
    runtime_monitor_sptr d_rtmon;
    std::vector<runtime_proxy_sptr> d_runtime_proxies;
};
} // namespace gr
