#pragma once 

#include <gnuradio/edge.hpp>

namespace gr {

template <class T>
static std::vector<T> unique_vector(std::vector<T> v)
{
    std::vector<T> result;
    std::insert_iterator<std::vector<T>> inserter(result, result.begin());

    sort(v.begin(), v.end());
    unique_copy(v.begin(), v.end(), inserter);
    return result;
}

/**
 * @brief Represents a set of ports connected by edges
 *
 */

class graph : public node, public std::enable_shared_from_this<graph>
{
protected:
    node_vector_t _nodes;
    edge_vector_t _edges;
    node_vector_t _orphan_nodes;

public:
    typedef std::shared_ptr<graph> sptr;
    static sptr make() { return std::make_shared<graph>(); }
    graph() : node() {}
    ~graph() {}
    std::shared_ptr<graph> base() { return shared_from_this(); }
    edge_vector_t& edges() { return _edges; }
    node_vector_t& orphan_nodes() { return _orphan_nodes; }
    node_vector_t& nodes() { return _nodes; }
    edge_sptr connect(const node_endpoint& src,
                 const node_endpoint& dst);
    edge_sptr connect(node_sptr src_node,
                 unsigned int src_port_index,
                 node_sptr dst_node,
                 unsigned int dst_port_index);
    edge_sptr connect(node_sptr src_node,
                 const std::string& src_port_name,
                 node_sptr dst_node,
                 const std::string& dst_port_name);
    void disconnect(const node_endpoint& src, const node_endpoint& dst){};
    virtual void validate(){};
    virtual void clear(){};
    void add_orphan_node(node_sptr orphan_node);
    void add_edge(edge_sptr edge);

    // }
    node_vector_t calc_used_nodes();
    edge_vector_t find_edge(port_sptr port);
};

typedef std::shared_ptr<graph> graph_sptr;


} // namespace gr
