#pragma once

#include "api.h"
#include <gnuradio/block.hpp>
#include <gnuradio/buffer.hpp>
#include <gnuradio/node.hpp>
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
    endpoint(){};
    virtual ~endpoint(){};
    endpoint(A a, B b) : std::pair<A, B>(a, b) {}
};

class node_endpoint : public endpoint<node_sptr, port_sptr>
{
private:
    node_sptr d_node;
    port_sptr d_port;

public:
    node_endpoint() : endpoint(){};
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

    ~node_endpoint(){};
    node_sptr node() const { return d_node; }
    port_sptr port() const { return d_port; }
    std::string identifier() const { return d_node->alias() + ":" + d_port->alias(); };
};

inline bool operator==(const node_endpoint& n1, const node_endpoint& n2)
{
    return (n1.node() == n2.node() && n1.port() == n2.port());
}

inline std::ostream& operator<<(std::ostream& os, const node_endpoint endp)
{
    os << endp.identifier();
    return os;
}


class edge
{
protected:
    node_endpoint _src, _dst;
    buffer_factory_function _buffer_factory = nullptr;
    std::shared_ptr<buffer_properties> _buffer_properties = nullptr;

public:
    edge(){};
    edge(const node_endpoint& src,
         const node_endpoint& dst,
         buffer_factory_function buffer_factory_ = nullptr,
         std::shared_ptr<buffer_properties> buffer_properties_ = nullptr)
        : _src(src),
          _dst(dst),
          _buffer_factory(buffer_factory_),
          _buffer_properties(buffer_properties_)
    {
    }
    edge(node_sptr src_blk,
         port_sptr src_port,
         node_sptr dst_blk,
         port_sptr dst_port,
         buffer_factory_function buffer_factory_ = nullptr,
         std::shared_ptr<buffer_properties> buffer_properties_ = nullptr)
        : _src(node_endpoint(src_blk, src_port)),
          _dst(node_endpoint(dst_blk, dst_port)),
          _buffer_factory(buffer_factory_),
          _buffer_properties(buffer_properties_)
    {
    }
    virtual ~edge(){};
    node_endpoint src() const { return _src; }
    node_endpoint dst() const { return _dst; }

    std::string identifier() const
    {
        return _src.identifier() + "->" + _dst.identifier();
    }

    size_t itemsize() const { return _src.port()->itemsize(); }

    bool has_custom_buffer() { 
        return _buffer_factory != nullptr; 
    }
    buffer_factory_function buffer_factory() { return _buffer_factory; }
    std::shared_ptr<buffer_properties> buf_properties() { return _buffer_properties; }
};

inline std::ostream& operator<<(std::ostream& os, const edge edge)
{
    os << edge.identifier();
    return os;
}

inline bool operator==(const edge& e1, const edge& e2)
{
    return (e1.src() == e2.src() && e1.dst() == e2.dst());
}

typedef std::vector<edge> edge_vector_t;
typedef std::vector<edge>::iterator edge_viter_t;

} // namespace gr
