
#ifndef INCLUDED_GR_RUNTIME_GRAPH_H
#define INCLUDED_GR_RUNTIME_GRAPH_H

#include "api.h"
#include <gnuradio/blocklib/block.hpp>
#include <gnuradio/blocklib/node.hpp>
#include <iostream>
#include <utility>
#include <vector>

namespace gr {

// class flat_graph;
template <class T>
static std::vector<T> unique_vector(std::vector<T> v)
{
    std::vector<T> result;
    std::insert_iterator<std::vector<T>> inserter(result, result.begin());

    sort(v.begin(), v.end());
    unique_copy(v.begin(), v.end(), inserter);
    return result;
}

// typedef endpoint std::pair<node_sptr, port_sptr>
template <class A, class B>
class endpoint : public std::pair<A, B>
{
private:
    A _a;
    B _b;

public:
    endpoint() {};
    virtual ~endpoint() {};
    endpoint(A a, B b) : std::pair<A, B>(a, b) {}
};

class node_endpoint : public endpoint<node_sptr, port_sptr>
{
private:
    node_sptr d_node;
    port_sptr d_port;

public:
    node_endpoint() : endpoint() {};
    node_endpoint(node_sptr node, port_sptr port)
        : endpoint<node_sptr, port_sptr>(node, port), d_node(node), d_port(port)
    {
    }

    node_endpoint(const node_endpoint& n)
        : endpoint<node_sptr, port_sptr>(n.node(), n.port())
    {
        d_node = n.node();
        d_port = n.port();
    }

    ~node_endpoint() {};
    node_sptr node() const { return d_node; }
    port_sptr port() const { return d_port; }
    std::string identifier() const { return d_node->alias() + ":" + d_port->alias(); };
};

inline std::ostream& operator<<(std::ostream& os, const node_endpoint endp)
{
    os << endp.identifier();
    return os;
}


// class block_endpoint : node_endpoint
// {
// private:
//     block_sptr d_block;

// public:
//     node_endpoint(block_sptr block, port_sptr port)
//         : node_endpoint(block, port), d_block(node)
//     {
//     }
// };

class edge
{
protected:
    node_endpoint _src, _dst;

public:
    edge(){};
    edge(const node_endpoint& src, const node_endpoint& dst) : _src(src), _dst(dst)
    {

    }
    edge(node_sptr src_blk, port_sptr src_port, node_sptr dst_blk, port_sptr dst_port)
        : _src(node_endpoint(src_blk, src_port)), _dst(node_endpoint(dst_blk, dst_port))
    {
    }
    virtual ~edge() {};
    node_endpoint src() { return _src; }
    node_endpoint dst() { return _dst; }

    std::string identifier() const
    {
        return _src.identifier() + "->" + _dst.identifier();
    }

    size_t itemsize() const { return _src.port()->itemsize(); }
};

inline std::ostream& operator<<(std::ostream& os, const edge edge)
{
    os << edge.identifier();
    return os;
}

typedef std::vector<edge> edge_vector_t;
typedef std::vector<edge>::iterator edge_viter_t;


/**
 * @brief Represents a set of ports connected by edges
 *
 */

class graph : public node, public std::enable_shared_from_this<graph>
{
protected:
    node_vector_t _nodes;
    edge_vector_t _edges;

public:
    graph() : node() {}
    ~graph() {}
    std::shared_ptr<graph> base() { return shared_from_this(); }
    std::vector<edge>& edges() { return _edges; }
    void connect(const node_endpoint& src, const node_endpoint& dst)
    {
        // TODO: Do a bunch of checking

        _edges.push_back(edge(src, dst));
        _nodes = calc_used_nodes();

        // update the block alias
        for (auto &b : _nodes)
        {
            std::map<std::string, int> name_count;
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
            name_count[b->name()] = cnt+1;
        }
    }
    void connect(node_sptr src_node,
                 unsigned int src_port_index,
                 node_sptr dst_node,
                 unsigned int dst_port_index)
    {
        port_sptr src_port = src_node->get_port(src_port_index, port_type_t::STREAM, port_direction_t::OUTPUT);
        if (src_port == nullptr)
            throw std::invalid_argument("Source Port not found");

        port_sptr dst_port = dst_node->get_port(dst_port_index, port_type_t::STREAM, port_direction_t::INPUT);
        if (dst_port == nullptr)
            throw std::invalid_argument("Destination port not found");

        connect(node_endpoint(src_node, src_port), node_endpoint(dst_node, dst_port));
    }
    void connect(node_sptr src_node,
                 std::string& src_port_name,
                 node_sptr dst_node,
                 std::string& dst_port_name) {};
    void disconnect(const node_endpoint& src, const node_endpoint& dst) {};
    virtual void validate() {};
    virtual void clear() {};

    // /**
    //  * @brief Return a flattened graph (all subgraphs reduced to their constituent blocks
    //  * and edges)
    //  *
    //  * @return graph
    //  */
    // flat_graph_sptr flatten()
    // {
    //     // For now we assume the graph is already flattened and we just downcast everything

        

    // }
    node_vector_t calc_used_nodes()
    {
        node_vector_t tmp;

        // Collect all blocks in the edge list
        for (edge_viter_t p = _edges.begin(); p != _edges.end(); p++) {
            tmp.push_back(p->src().node());
            tmp.push_back(p->dst().node());
        }

        return unique_vector<node_sptr>(tmp);
    }

};

typedef std::shared_ptr<graph> graph_sptr;


} // namespace gr
#endif