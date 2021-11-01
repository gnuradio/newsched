#pragma once

#include <gnuradio/flowgraph_monitor.hh>
#include <gnuradio/graph.hh>
#include <gnuradio/scheduler.hh>
#include <gnuradio/domain.hh>

namespace gr {

/**
 * @brief Top level graph representing the flowgraph 
 * 
 */
class flowgraph : public graph
{
private:
    std::vector<scheduler_sptr> d_schedulers;
    flat_graph_sptr d_flat_graph;
    std::vector<graph_sptr> d_subgraphs;
    std::vector<flat_graph_sptr> d_flat_subgraphs;

    bool _monitor_thread_stopped = false;
    flowgraph_monitor_sptr d_fgmon;
    bool _validated = false; // TODO - update when connections are added or things are changed

    // Dynamically Loaded Default Scheduler
    const std::string s_default_scheduler_name = "nbt";
    scheduler_sptr d_default_scheduler = nullptr;
    bool d_default_scheduler_inuse = true;

public:
    
    typedef std::shared_ptr<flowgraph> sptr;
    static sptr make(const std::string& name = "flowgraph") { return std::make_shared<flowgraph>(name); }
    flowgraph(const std::string& name = "flowgraph");
    virtual ~flowgraph() { _monitor_thread_stopped = true; };
    void set_scheduler(scheduler_sptr sched);
    void set_schedulers(std::vector<scheduler_sptr> sched);
    void add_scheduler(scheduler_sptr sched);
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
