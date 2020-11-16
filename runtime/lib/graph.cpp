#include <gnuradio/graph.hpp>

namespace gr {

void graph::connect(const node_endpoint& src,
                    const node_endpoint& dst,
                    buffer_factory_function buffer_factory,
                    std::shared_ptr<buffer_properties> buf_properties)
{
    // TODO: Do a bunch of checking

    _edges.push_back(edge(src, dst, buffer_factory, buf_properties));
    auto used_nodes = calc_used_nodes();
    // used_nodes.insert(used_nodes.end(), _orphan_nodes.begin(),
    // _orphan_nodes.end());
    _nodes = used_nodes;

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
        b->set_alias(b->name() + std::to_string(cnt));
        name_count[b->name()] = cnt + 1;
    }
}
void graph::connect(node_sptr src_node,
                    unsigned int src_port_index,
                    node_sptr dst_node,
                    unsigned int dst_port_index,
                    buffer_factory_function buffer_factory,
                    std::shared_ptr<buffer_properties> buf_properties)
{
    port_sptr src_port =
        src_node->get_port(src_port_index, port_type_t::STREAM, port_direction_t::OUTPUT);
    if (src_port == nullptr)
        throw std::invalid_argument("Source Port not found");

    port_sptr dst_port =
        dst_node->get_port(dst_port_index, port_type_t::STREAM, port_direction_t::INPUT);
    if (dst_port == nullptr)
        throw std::invalid_argument("Destination port not found");

    connect(node_endpoint(src_node, src_port),
            node_endpoint(dst_node, dst_port),
            buffer_factory,
            buf_properties);
}

void graph::add_orphan_node(node_sptr orphan_node)
{
    _orphan_nodes.push_back(orphan_node);
}

node_vector_t graph::calc_used_nodes()
{
    node_vector_t tmp;

    // Collect all blocks in the edge list
    for (edge_viter_t p = _edges.begin(); p != _edges.end(); p++) {
        tmp.push_back(p->src().node());
        tmp.push_back(p->dst().node());
    }
    for (auto n : _orphan_nodes) {
        tmp.push_back(n);
    }

    return unique_vector<node_sptr>(tmp);
}

std::vector<edge> graph::find_edge(port_sptr port)
{
    std::vector<edge> ret;
    for (auto& e : edges()) {
        if (e.src().port() == port)
            ret.push_back(e);

        if (e.dst().port() == port)
            ret.push_back(e);
    }

    if (ret.empty())
        throw std::invalid_argument("edge not found");

    return ret;
}


} // namespace gr