#pragma once

#include <gnuradio/flowgraph_monitor.hpp>
#include <gnuradio/graph.hpp>
#include <gnuradio/scheduler.hpp>
#include <gnuradio/domain.hpp>

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

public:
    
    typedef std::shared_ptr<flowgraph> sptr;
    static sptr make() { return std::make_shared<flowgraph>(); }
    flowgraph() { set_alias("flowgraph"); };
    virtual ~flowgraph() { _monitor_thread_stopped = true; };
    void set_scheduler(scheduler_sptr sched);
    void set_schedulers(std::vector<scheduler_sptr> sched);
    void add_scheduler(scheduler_sptr sched);
    void clear_schedulers();
    void partition(std::vector<domain_conf>& confs);
    void validate();
    void start();
    void stop();
    void wait();
    void run();
};

typedef flowgraph::sptr flowgraph_sptr;
} // namespace gr
