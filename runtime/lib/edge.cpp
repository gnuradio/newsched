#include <gnuradio/edge.hpp>

namespace gr {

node_endpoint::node_endpoint() : endpoint(){};
node_endpoint::node_endpoint(node_sptr node, port_sptr port)
    : endpoint<node_sptr, port_sptr>(node, port), d_node(node), d_port(port)
{
}

node_endpoint::node_endpoint(const node_endpoint& n)
    : endpoint<node_sptr, port_sptr>(n.node(), n.port())
{
    d_node = n.node();
    d_port = n.port();
}

node_sptr node_endpoint::node() const { return d_node; }
port_sptr node_endpoint::port() const { return d_port; }
std::string node_endpoint::identifier() const
{
    return d_node->alias() + ":" + d_port->alias();
};

edge::edge(){};
edge::edge(const node_endpoint& src,
           const node_endpoint& dst,
           buffer_factory_function buffer_factory_,
           std::shared_ptr<buffer_properties> buffer_properties_)
    : _src(src),
      _dst(dst),
      _buffer_factory(buffer_factory_),
      _buffer_properties(buffer_properties_)
{
}
edge::edge(node_sptr src_blk,
           port_sptr src_port,
           node_sptr dst_blk,
           port_sptr dst_port,
           buffer_factory_function buffer_factory_,
           std::shared_ptr<buffer_properties> buffer_properties_)
    : _src(node_endpoint(src_blk, src_port)),
      _dst(node_endpoint(dst_blk, dst_port)),
      _buffer_factory(buffer_factory_),
      _buffer_properties(buffer_properties_)
{
}

node_endpoint edge::src() const { return _src; }
node_endpoint edge::dst() const { return _dst; }

std::string edge::identifier() const
{
    return _src.identifier() + "->" + _dst.identifier();
}

size_t edge::itemsize() const { return _src.port()->itemsize(); }

bool edge::has_custom_buffer() { return _buffer_factory != nullptr; }
buffer_factory_function edge::buffer_factory() { return _buffer_factory; }
std::shared_ptr<buffer_properties> edge::buf_properties() { return _buffer_properties; }


} // namespace gr
