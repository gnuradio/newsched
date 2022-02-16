#pragma once

#include <gnuradio/graph.hh>
#include <gnuradio/flat_graph.hh>

namespace gr {

/**
 * @brief Top level graph representing the flowgraph 
 * 
 */
class runtime;
class flowgraph : public graph
{
public:
    
    typedef std::shared_ptr<flowgraph> sptr;
    static sptr make(const std::string& name = "flowgraph") { return std::make_shared<flowgraph>(name); }
    flowgraph(const std::string& name = "flowgraph");
    virtual ~flowgraph() { };
    static void check_connections(const graph_sptr& g);
    flat_graph_sptr make_flat();

    void start();
    void stop();
    void wait();
    void run();

private:
    std::shared_ptr<gr::runtime> d_runtime_sptr = nullptr;

};

typedef flowgraph::sptr flowgraph_sptr;
} // namespace gr
