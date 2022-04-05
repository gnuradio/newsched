#pragma once

#include <gnuradio/graph.h>

namespace gr {
class hier_block : public graph
{
private:
    std::vector<edge_sptr> _input_edges;
    std::vector<edge_sptr> _output_edges;

public:
    // add_port becomes public only for hier_block
    void add_port(port_sptr p) { graph::add_port(p); }
    edge_sptr connect(const node_endpoint& src, const node_endpoint& dst) override
    {
        if (src.node().get() == this) { // connecting to self input ports
            _input_edges.push_back(edge::make(src, dst));
        }
        else if (dst.node().get() == this) { // connection to self output ports
            _output_edges.push_back(edge::make(src, dst));
        }
        else // internal connection
        {
            return graph::connect(src, dst);
        }

        return nullptr; // graph::connect(src, dst);
    }


    edge_sptr connect(node_sptr src_node,
                      unsigned int src_port_index,
                      node_sptr dst_node,
                      unsigned int dst_port_index) override
    {
        auto src_direction = port_direction_t::OUTPUT;
        auto dst_direction = port_direction_t::INPUT;
        if (src_node.get() == this) {
            src_direction = port_direction_t::INPUT;
        }

        if (dst_node.get() == this) {
            dst_direction = port_direction_t::OUTPUT;
        }

        port_sptr src_port =
            (src_node == nullptr)
                ? nullptr
                : src_node->get_port(src_port_index,
                                     port_type_t::STREAM,
                                     src_direction); // for hier block it is reversed

        port_sptr dst_port =
            (dst_node == nullptr)
                ? nullptr
                : dst_node->get_port(dst_port_index, port_type_t::STREAM, dst_direction);

        return connect(node_endpoint(src_node, src_port),
                       node_endpoint(dst_node, dst_port));
    }

    std::vector<edge_sptr>& input_edges() { return _input_edges; }
    std::vector<edge_sptr>& output_edges() { return _output_edges; }
};
} // namespace gr