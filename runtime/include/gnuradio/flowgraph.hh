#pragma once

#include <gnuradio/domain.hh>
#include <gnuradio/fgm_proxy.hh>
#include <gnuradio/flowgraph_monitor.hh>
#include <gnuradio/graph.hh>
#include <gnuradio/scheduler.hh>

namespace gr {

/**
 * @brief Top level graph representing the flowgraph
 *
 */
class flowgraph : public graph
{
private:
    std::vector<scheduler_sptr> d_schedulers;
    std::vector<fgm_proxy_sptr> d_fgm_proxies;
    flat_graph_sptr d_flat_graph;
    std::vector<flat_graph_sptr> d_flat_subgraphs;

    bool _monitor_thread_stopped = false;
    flowgraph_monitor_sptr d_fgmon;
    bool _validated =
        false; // TODO - update when connections are added or things are changed

    // Dynamically Loaded Default Scheduler
    const std::string s_default_scheduler_name = "nbt";
    const bool s_secondary = false;
    scheduler_sptr d_default_scheduler = nullptr;
    bool d_default_scheduler_inuse = true;

public:
    typedef std::shared_ptr<flowgraph> sptr;
    static sptr make(const std::string& name = "flowgraph", bool secondary = false)
    {
        return std::make_shared<flowgraph>(name, secondary);
    }
    flowgraph(const std::string& name = "flowgraph", bool secondary = false);
    virtual ~flowgraph() { _monitor_thread_stopped = true; };
    void set_scheduler(scheduler_sptr sched);
    void set_schedulers(std::vector<scheduler_sptr> sched);
    void add_scheduler(scheduler_sptr sched);
    void add_fgm_proxy(fgm_proxy_sptr fgm_proxy);
    // fgm_proxy_sptr add_upstream_proxy(const std::string& ipaddr, int port)
    // {
    //     auto proxy_obj = std::make_shared<fgm_proxy>(d_fgmon, ipaddr, port, true);
        
    //     return proxy_obj;
    // }
    // fgm_proxy_sptr add_downstream_proxy(const std::string& ipaddr, int port)
    // {
    //     return std::make_shared<fgm_proxy>(d_fgmon, ipaddr, port, false);
    // }
    void clear_schedulers();
    void partition(std::vector<domain_conf>& confs);
    void check_connections(const graph_sptr& g);
    void validate();
    void start();
    void stop();
    void wait();
    void run();
};

typedef flowgraph::sptr flowgraph_sptr;
} // namespace gr
