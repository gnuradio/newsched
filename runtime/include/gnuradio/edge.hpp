#pragma once

#include <gnuradio/buffer.hpp>
#include <gnuradio/node.hpp>
#include <iostream>
#include <utility>
#include <vector>

namespace gr {

/**
 * @brief Wrapper of pair for describing general endpoint between two things
 *
 * @tparam A
 * @tparam B
 */
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

/**
 * @brief Endpoint between ports associated with nodes
 *
 */
class node_endpoint : public endpoint<node_sptr, port_sptr>
{
private:
    node_sptr d_node;
    port_sptr d_port;

public:
    node_endpoint();
    node_endpoint(node_sptr node, port_sptr port);
    node_endpoint(const node_endpoint& n);

    ~node_endpoint(){};
    node_sptr node() const;
    port_sptr port() const;
    std::string identifier() const;
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

/**
 * @brief Edge between ports
 *
 * The edge class stores the properties describing the logical connection between two
 * ports.  This includes the buffer used to pass data between the ports, which may be of a
 * custom type not defined in-tree
 *
 */
class edge
{
protected:
    node_endpoint _src, _dst;
    buffer_factory_function _buffer_factory = nullptr;
    std::shared_ptr<buffer_properties> _buffer_properties = nullptr;

public:
    typedef std::shared_ptr<edge> sptr;
    static sptr make(const node_endpoint& src, const node_endpoint& dst)
    {
        return std::make_shared<edge>(src, dst);
    }
    static sptr
    make(node_sptr src_blk, port_sptr src_port, node_sptr dst_blk, port_sptr dst_port)
    {
        return std::make_shared<edge>(src_blk, src_port, dst_blk, dst_port);
    }
    edge(const node_endpoint& src, const node_endpoint& dst);
    edge(node_sptr src_blk, port_sptr src_port, node_sptr dst_blk, port_sptr dst_port);
    virtual ~edge(){};
    node_endpoint src() const;
    node_endpoint dst() const;

    std::string identifier() const;
    size_t itemsize() const;

    void set_custom_buffer(buffer_factory_function buffer_factory,
                           std::shared_ptr<buffer_properties> buffer_properties = nullptr)
    {
        _buffer_factory = buffer_factory;
        _buffer_properties = buffer_properties;
    }

    bool has_custom_buffer();
    buffer_factory_function buffer_factory();
    std::shared_ptr<buffer_properties> buf_properties();
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

typedef edge::sptr edge_sptr;
typedef std::vector<edge_sptr> edge_vector_t;

} // namespace gr
