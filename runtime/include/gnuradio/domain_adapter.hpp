#pragma once

#include <zmq.hpp>
#include <thread>

#include <gnuradio/node.hpp>
#include <gnuradio/buffer.hpp>
#include <gnuradio/graph.hpp>

namespace gr {


enum class buffer_location_t { LOCAL = 0, REMOTE };
enum class buffer_preference_t { UPSTREAM, DOWNSTREAM };


enum class da_request_t : uint32_t {
    CANCEL = 0,
    WRITE_INFO,
    READ_INFO,
    POST_WRITE,
    POST_READ,
    GET_REMOTE_BUFFER
};

enum class da_response_t : uint32_t { OK = 0, ERROR = 1 };

/**
 * @brief Domain Adapter used internally by flowgraphs to handle domain crossings
 *
 * The Domain Adapter is both a node in that it is connected to blocks at the edges of a
 * subgraph as well as a buffer, since it is used for the scheduler to get the address
 * needed to read from or write to
 *
 * It holds a pointer to a buffer object which may be null if the adapter is not hosting
 * the buffer and relying on its peer to host the buffer
 */
class domain_adapter : public node, public buffer
{
protected:
    buffer_sptr _buffer = nullptr;
    buffer_location_t _buffer_loc;

    domain_adapter(buffer_location_t buf_loc)
        : node("domain_adapter"), _buffer_loc(buf_loc)
    {
    }

public:
    void set_buffer(buffer_sptr buf) { _buffer = buf; }
    buffer_sptr buffer() { return _buffer; }

    buffer_location_t buffer_location() { return _buffer_loc; }
    void set_buffer_location(buffer_location_t buf_loc) { _buffer_loc = buf_loc; }
};

typedef std::shared_ptr<domain_adapter> domain_adapter_sptr;


// Domain configuration
class domain_adapter_conf
{
protected:
    domain_adapter_conf(buffer_preference_t buf_pref) : _buf_pref(buf_pref) {}
    buffer_preference_t _buf_pref;

public:
    virtual std::pair<domain_adapter_sptr, domain_adapter_sptr>
    make_domain_adapter_pair(port_sptr upstream_port, port_sptr downstream_port)
    {
        throw std::runtime_error("Cannot create domain adapter pair from base class");
    };
};

typedef std::shared_ptr<domain_adapter_conf> domain_adapter_conf_sptr;
typedef std::vector<std::tuple<edge, domain_adapter_conf_sptr>>
    domain_adapter_conf_per_edge;



} // namespace gr
