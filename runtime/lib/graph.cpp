#include <gnuradio/graph.hpp>

namespace gr {

edge_sptr graph::connect(const node_endpoint& src,
                    const node_endpoint& dst)
{
    
    if (src.port()->itemsize() != dst.port()->itemsize())
    {
        std::stringstream msg;
        msg << "itemsize mismatch: " << src << " using " << src.port()->itemsize() << ", " << dst
            << " using " << dst.port()->itemsize();
        throw std::invalid_argument(msg.str());
    }

    // If not untyped ports, check that data types are the same
    // TODO
    
    auto newedge = edge::make(src, dst);
    _edges.push_back(newedge);
    _nodes = calc_used_nodes();

    std::map<std::string, int> name_count;
    // update the block alias
    for (auto& b : _nodes) {
        // look in the map, see how many of that name exist
        // make the alias name + count;
        // increment the map
        int cnt;
        if (name_count.find(b->name()) == name_count.end()) {
            name_count[b->name()] = cnt = 0;
        } else {
            cnt = name_count[b->name()];
        }
        //b->set_alias(b->name() + std::to_string(cnt));
        name_count[b->name()] = cnt + 1;

        // for now, just use the name+nodeid as the alias
        b->set_alias(b->name() + "(" + std::to_string(b->id()) + ")");
    }

    // Give the underlying port objects information about the connected ports
    src.port()->connect(dst.port());
    dst.port()->connect(src.port());

    return newedge;
}

edge_sptr graph::connect(node_sptr src_node,
                    unsigned int src_port_index,
                    node_sptr dst_node,
                    unsigned int dst_port_index)
{
    port_sptr src_port =
        src_node->get_port(src_port_index, port_type_t::STREAM, port_direction_t::OUTPUT);
    if (src_port == nullptr)
        throw std::invalid_argument("Source Port not found");

    port_sptr dst_port =
        dst_node->get_port(dst_port_index, port_type_t::STREAM, port_direction_t::INPUT);
    if (dst_port == nullptr)
        throw std::invalid_argument("Destination port not found");

    return connect(node_endpoint(src_node, src_port),
            node_endpoint(dst_node, dst_port));
}

edge_sptr graph::connect(node_sptr src_node,
                    const std::string& src_port_name,
                    node_sptr dst_node,
                    const std::string& dst_port_name)
{
    port_sptr src_port =
        src_node->get_port(src_port_name);
    if (src_port == nullptr)
        throw std::invalid_argument("Source Port not found");

    port_sptr dst_port =
        dst_node->get_port(dst_port_name);
    if (dst_port == nullptr)
        throw std::invalid_argument("Destination port not found");

    return connect(node_endpoint(src_node, src_port),
            node_endpoint(dst_node, dst_port));
}


void graph::add_orphan_node(node_sptr orphan_node)
{
    _orphan_nodes.push_back(orphan_node);
}

node_vector_t graph::calc_used_nodes()
{
    node_vector_t tmp;

    // Collect all blocks in the edge list
    for (auto& p : edges()) {
        tmp.push_back(p->src().node());
        tmp.push_back(p->dst().node());
    }
    for (auto n : _orphan_nodes) {
        tmp.push_back(n);
    }

    return unique_vector<node_sptr>(tmp);
}

edge_vector_t graph::find_edge(port_sptr port)
{
    edge_vector_t ret;
    for (auto& e : edges()) {
        if (e->src().port() == port)
            ret.push_back(e);

        if (e->dst().port() == port)
            ret.push_back(e);
    }

    // TODO: check optional flag
    // msg ports or optional streaming ports might not be connected
    // if (ret.empty())
    //     throw std::invalid_argument("edge not found");

    return ret;
}

void graph::add_edge(edge_sptr edge)
{
    // TODO: check that edge is not already in the graph
    _edges.push_back(edge);
}

} // namespace gr
