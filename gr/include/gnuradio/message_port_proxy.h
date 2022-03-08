#pragma once

#include <gnuradio/logger.h>
#include <gnuradio/port_interface.h>
#include <gnuradio/scheduler_message.h>
#include <zmq.hpp>
#include <thread>

namespace gr {
class message_port_proxy_upstream : public port_interface
{
public:
    using sptr = std::shared_ptr<message_port_proxy_upstream>;
    static sptr make() { return std::make_shared<message_port_proxy_upstream>(); }
    message_port_proxy_upstream() : _context(1), _socket(_context, zmq::socket_type::push)
    {
        _socket.set(zmq::sockopt::sndhwm, 1);
        _socket.set(zmq::sockopt::rcvhwm, 1);
    }
    ~message_port_proxy_upstream() override
    {
        _context.shutdown();
        _socket.close();
        _context.close();
    }

    void push_message(scheduler_message_sptr msg) override
    {
        _socket.send(zmq::buffer(msg->to_json()), zmq::send_flags::none);
    }

    void connect(const std::string& ipaddr, int port)
    {
        std::string cliendpoint = "tcp://" + ipaddr + ":" + std::to_string(port);
        std::cout << "connecting: " << cliendpoint << std::endl;
        _socket.connect(cliendpoint);

        _port = port;
        _connected = true;
    }

private:
    zmq::context_t _context;
    zmq::socket_t _socket;
    int _port;
    bool _connected = false;
    int _id;

    logger_ptr d_logger, debug_logger;
    bool _rcv_done = false;
};

class message_port_proxy_downstream : public port_interface
{
public:
    using sptr = std::shared_ptr<message_port_proxy_downstream>;
    static sptr make(int port)
    {
        return std::make_shared<message_port_proxy_downstream>(port);
    }
    message_port_proxy_downstream(int port)
        : _context(1), _socket(_context, zmq::socket_type::pull), _port(port)
    {
        gr::configure_default_loggers(d_logger, d_debug_logger, "message_proxy_downstream");
        _socket.set(zmq::sockopt::sndhwm, 1);
        _socket.set(zmq::sockopt::rcvhwm, 1);

        std::string svrendpoint = "tcp://*:" + std::to_string(_port);
        std::cout << "binding " << svrendpoint << std::endl;
        _socket.bind(svrendpoint);

        if (_port == 0) {
            std::string endpoint_str = _socket.get(zmq::sockopt::last_endpoint);
            auto colon_index = endpoint_str.find_last_of(':');
            std::string port_str =
                std::string(endpoint_str.begin() + colon_index + 1, endpoint_str.end());
            _port = std::stoi(port_str);
        }
    }
    ~message_port_proxy_downstream() override
    {
        _context.shutdown();
        _socket.close();
        _context.close();
    }
    int port() { return _port; }
    void start_rx()
    {
        std::thread t([this]() {
            d_logger->info("Client is connected, starting recv loop");
            while (!this->_rcv_done) {
                d_logger->info("top");
                try {
                    _rcv_msg.rebuild();
                    d_logger->info("Going into recv");
                    if (auto res = _socket.recv(_rcv_msg)) //, zmq::recv_flags::dontwait))
                    {
                        auto str = _rcv_msg.to_string();
                        // convert to scheduler message
                        msgport_message m;
                        auto msg = m.from_json(str);
                        _grport->push_message(msg);
                    }
                    else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    }
                } catch (zmq::error_t const& err) {
                    d_logger->info("Caught error");
                    this->_rcv_done = true;
                }
            }
        });
        t.detach();
    }
    void set_gr_port(port_interface_sptr p) { _grport = p; };
    void push_message(scheduler_message_sptr msg) override
    {
        throw std::runtime_error(
            "push_message should not be called from downstream proxy");
    }

private:
    port_interface_sptr _grport = nullptr;

    zmq::context_t _context;
    zmq::socket_t _socket;
    zmq::message_t _rcv_msg;
    int _port;
    bool _connected = false;
    int _id;

    logger_ptr d_logger, d_debug_logger;
    bool _rcv_done = false;
};

} // namespace gr