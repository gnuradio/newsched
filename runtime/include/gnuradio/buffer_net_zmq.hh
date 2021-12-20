#pragma once

#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <zmq.hpp>

#include <gnuradio/buffer.hh>

namespace gr {


class buffer_net_zmq_reader;

class buffer_net_zmq : public buffer
{
private:
    std::vector<uint8_t> _buffer;

    zmq::context_t _context;
    zmq::socket_t _socket;



public:
    typedef std::shared_ptr<buffer_net_zmq> sptr;
    buffer_net_zmq(size_t num_items,
                   size_t item_size,
                   std::shared_ptr<buffer_properties> buffer_properties,
                   int port);
    virtual ~buffer_net_zmq(){};
    static buffer_sptr make(size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties);

    void* read_ptr(size_t index) {
        return nullptr;
    }
    virtual size_t space_available() override
    {
        return _num_items;
    }
    virtual void* write_ptr() override {
        return _buffer.data();
    }

    virtual void post_write(int num_items) override {
        // send the data from buffer over the socket
        GR_LOG_DEBUG(_debug_logger, "sending {} items", num_items);
        auto res = _socket.send(zmq::buffer(write_ptr(), num_items * _item_size), zmq::send_flags::none);
        GR_LOG_DEBUG(_debug_logger, "send returned code {}", *res);
    }

    virtual std::shared_ptr<buffer_reader>
    add_reader(std::shared_ptr<buffer_properties> buf_props, size_t itemsize) override
    {
        // do nothing because readers will be added on zmq connect
        return nullptr;
    }
};

class buffer_net_zmq_reader : public buffer_reader
{
private:
    zmq::context_t _context;
    zmq::socket_t _socket;  
    zmq::message_t _msg;
    size_t _msg_idx = 0;
    size_t _msg_size = 0;

    // Circular buffer for zmq to write into
    gr::buffer_sptr _circbuf;
    gr::buffer_reader_sptr _circbuf_rdr;

    logger_sptr _logger;
    logger_sptr _debug_logger;

public:
    bool _recv_done = false;
    static buffer_reader_sptr make(size_t itemsize,
                                   std::shared_ptr<buffer_properties> buf_props);
    buffer_net_zmq_reader(
                          std::shared_ptr<buffer_properties> buf_props,
                          size_t itemsize,
                                   const std::string& ipaddr,
                                   int port);

    virtual ~buffer_net_zmq_reader(){};

    virtual bool read_info(buffer_info_t& info) { 
        auto ret = _circbuf_rdr->read_info(info);
        return  ret;

    }
    void* read_ptr() { 
        return _circbuf_rdr->read_ptr(); 
    }

    // Tags not supported yet
    const std::vector<tag_t>& tags() const override { return _circbuf->tags(); }
    std::vector<tag_t> get_tags(size_t num_items) { return {}; }
    virtual void post_read(int num_items) {
        GR_LOG_DEBUG(_debug_logger, "post_read: {}", num_items);
        _circbuf_rdr->post_read(num_items);
    }
};


class buffer_net_zmq_properties : public buffer_properties
{
public:
    // typedef sptr std::shared_ptr<buffer_properties>;
    buffer_net_zmq_properties(const std::string& ipaddr, int port)
        : buffer_properties(), _ipaddr(ipaddr), _port(port)
    {
        _bff = buffer_net_zmq::make;
        _brff = buffer_net_zmq_reader::make;
    }
    virtual ~buffer_net_zmq_properties(){};

    static std::shared_ptr<buffer_properties> make(const std::string& ipaddr, int port)
    {
        return std::dynamic_pointer_cast<buffer_properties>(
            std::make_shared<buffer_net_zmq_properties>(ipaddr, port));
    }

    auto port() { return _port; }
    auto ipaddr() { return _ipaddr; }

private:
    std::string _ipaddr;
    int _port;
};

} // namespace gr
