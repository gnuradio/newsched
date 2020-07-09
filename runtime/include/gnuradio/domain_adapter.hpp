#pragma once

#include <zmq.hpp>


#include <gnuradio/blocklib/node.hpp>
#include <gnuradio/buffer.hpp>

namespace gr {


enum class buffer_location_t { LOCAL = 0, REMOTE, SERIALIZED };


enum class da_request_t : uint32_t {
    CANCEL = 0,
    WRITE_INFO,
    READ_INFO,
    POST_WRITE,
    POST_READ
};

/**
 * @brief Domain Adapter used internally by flowgraphs to handle domain crossings
 *
 */
class domain_adapter : public node, public buffer
{
protected:
    buffer_sptr _buffer = nullptr;
    buffer_location_t _buffer_loc;

    domain_adapter() : node("domain_adapter") {}

public:
    void set_buffer(buffer_sptr buf) { _buffer = buf; }
    buffer_sptr buffer() { return _buffer; }
};

/**
 * @brief Uses a simple zmq socket to communicate buffer pointers between domains
 *
 */
class domain_adapter_zmq_rep_svr : public domain_adapter
{
private:
    zmq::context_t* d_context;
    zmq::socket_t* d_socket;

    bool d_connected = false;
    std::thread d_thread;

public:
    typedef std::shared_ptr<domain_adapter_zmq_rep_svr> sptr;
    static sptr
    make(const std::string& endpoint_uri, buffer_location_t buf_loc, port_sptr other_port)
    {
        auto ptr = std::make_shared<domain_adapter_zmq_rep_svr>(
            domain_adapter_zmq_rep_svr(endpoint_uri, buf_loc));

        ptr->add_port(port_base::make("output",
                                      port_direction_t::OUTPUT,
                                      other_port->data_type(),
                                      port_type_t::STREAM,
                                      other_port->dims()));

        ptr->start_thread(ptr); // start thread with reference to shared pointer

        return ptr;
    }

    domain_adapter_zmq_rep_svr(const std::string& endpoint_uri, buffer_location_t buf_loc)
    {
        d_context = new zmq::context_t(1);
        d_socket = new zmq::socket_t(*d_context,
                                     zmq::socket_type::rep); // server is a rep socket
        int time = 0;
        d_socket->setsockopt(ZMQ_LINGER, &time, sizeof(time));
        d_socket->bind(endpoint_uri);
    }

    void start_thread(sptr ptr) { d_thread = std::thread(run_thread, ptr); }

    static void run_thread(sptr top) // zmq::socket_t* sock)
    {
        auto sock = top->d_socket;
        while (1) {
            zmq::message_t request(100);
            sock->recv(&request);

            // Parse the message
            auto action = *((da_request_t*)request.data());
            switch (action) {
            case da_request_t::CANCEL: {
                top->buffer()->cancel();
            } break;
            case da_request_t::WRITE_INFO: {
                buffer_info_t info = top->buffer()->write_info();


                zmq::message_t msg(sizeof(buffer_info_t));
                memcpy(msg.data(), &info, sizeof(buffer_info_t));
                sock->send(msg);
            } break;
            case da_request_t::POST_WRITE: {
                int num_items;
                memcpy(&num_items, (uint8_t*)request.data() + 4, sizeof(int));
                top->buffer()->post_write(num_items);


                zmq::message_t msg(0);
                sock->send(msg);
            } break;
            default: {
                // // send the reply to the client
                std::string str_msg("WORLD");
                // zmq::message_t msg(0);

                zmq::message_t msg(str_msg.begin(), str_msg.end());
                sock->send(msg);
            } break;
            }
        }
    }

    virtual void* read_ptr() { return nullptr; }
    virtual void* write_ptr() { return nullptr; }

    // virtual int capacity() = 0;
    // virtual int size() = 0;

    virtual buffer_info_t read_info() { return _buffer->read_info(); }
    virtual buffer_info_t write_info()
    {
        // should not get called
        throw std::runtime_error("write_info not valid for da_svr block"); // TODO logging
        buffer_info_t ret;
        return ret;
    }
    virtual void cancel() { _buffer->cancel(); }

    virtual void post_read(int num_items) { return _buffer->post_read(num_items); }
    virtual void post_write(int num_items) {}

    // This is not valid for all buffers, e.g. domain adapters
    virtual void copy_items(buffer_sptr from, int nitems) {}
};


class domain_adapter_zmq_req_cli : public domain_adapter
{
    zmq::context_t* d_context;
    zmq::socket_t* d_socket;

public:
    typedef std::shared_ptr<domain_adapter_zmq_req_cli> sptr;
    static sptr
    make(const std::string& endpoint_uri, buffer_location_t buf_loc, port_sptr other_port)
    {
        auto ptr = std::make_shared<domain_adapter_zmq_req_cli>(
            domain_adapter_zmq_req_cli(endpoint_uri, buf_loc));

        // Type of port is not known at compile time
        ptr->add_port(port_base::make("input",
                                      port_direction_t::INPUT,
                                      other_port->data_type(),
                                      port_type_t::STREAM,
                                      other_port->dims()));

        return ptr;
    }
    domain_adapter_zmq_req_cli(const std::string& endpoint_uri, buffer_location_t buf_loc)
    {
        d_context = new zmq::context_t(1);
        d_socket = new zmq::socket_t(*d_context,
                                     zmq::socket_type::req); // server is a rep socket
        int time = 0;
        d_socket->setsockopt(ZMQ_LINGER, &time, sizeof(time));
        d_socket->connect(endpoint_uri);
    }

    void test()
    {
        zmq::message_t response(100);
        std::string str_msg("hello");
        zmq::message_t msg(str_msg.begin(), str_msg.end());

        d_socket->send(msg);
        d_socket->recv(&response);

        std::string str =
            std::string(static_cast<char*>(response.data()), response.size());
        std::cout << str << std::endl;
    }

    virtual void* read_ptr() { return nullptr; }
    virtual void* write_ptr() { return nullptr; }

    // virtual int capacity() = 0;
    // virtual int size() = 0;

    virtual buffer_info_t read_info()
    {
        buffer_info_t ret;

        return ret;
    }
    virtual buffer_info_t write_info()
    {
        zmq::message_t msg(4);
        auto action = da_request_t::WRITE_INFO;
        memcpy(msg.data(), &action, 4);

        zmq::message_t response(sizeof(buffer_info_t));
        d_socket->send(msg);
        d_socket->recv(&response);

        buffer_info_t ret;
        memcpy(&ret, response.data(), sizeof(buffer_info_t));

        return ret;
    }
    virtual void cancel() {

        zmq::message_t msg(4);
        auto action = da_request_t::CANCEL;
        memcpy(msg.data(), &action, 4);

        zmq::message_t response(4);
        d_socket->send(msg);
        d_socket->recv(&response);


    }

    virtual void post_read(int num_items) {}
    virtual void post_write(int num_items)
    {
        zmq::message_t msg(4+sizeof(int));
        auto action = da_request_t::POST_WRITE;
        memcpy(msg.data(), &action, 4);
        memcpy((uint8_t*)msg.data() + 4, &num_items, sizeof(int));

        zmq::message_t response(0);
        d_socket->send(msg);
        d_socket->recv(&response);
    }

    // This is not valid for all buffers, e.g. domain adapters
    virtual void copy_items(buffer_sptr from, int nitems) {}
};

} // namespace gr