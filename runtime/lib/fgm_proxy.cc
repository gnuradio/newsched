#include <gnuradio/fgm_proxy.hh>
#include <gnuradio/flowgraph.hh>
#include <gnuradio/flowgraph_monitor.hh>
#include <chrono>
#include <thread>

namespace gr {
fgm_proxy::sptr fgm_proxy::make(const std::string& ipaddr, int port, bool upstream)
{
    return std::make_shared<fgm_proxy>(ipaddr, port, upstream);
}
fgm_proxy::fgm_proxy(const std::string& ipaddr, int port, bool upstream)
    : _upstream(upstream),
      _context(1),
      _server_socket(_context, zmq::socket_type::pull),
      _client_socket(_context, zmq::socket_type::push)

{
    _logger = logging::get_logger("fgm_proxy", "debug");
    int sndhwm = 1;
    int rcvhwm = 1;
    _server_socket.setsockopt(ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
    _server_socket.setsockopt(ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));

    _client_socket.setsockopt(ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
    _client_socket.setsockopt(ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));

    std::string svrendpoint = "tcp://*:" + std::to_string(upstream ? port : port + 1);
    std::cout << "binding " << svrendpoint;
    _server_socket.bind(svrendpoint);

    std::string cliendpoint =
        "tcp://" + ipaddr + ":" + std::to_string(upstream ? port + 1 : port);
    std::cout << "connecting " << cliendpoint;
    _client_socket.connect(cliendpoint);

    std::cout << "connected." << std::endl;

    std::thread t([this]() {
        while (!this->_rcv_done) { // (!this->_recv_done) {
            try {
                _rcv_msg.rebuild();
                if (auto res =
                        _server_socket.recv(_rcv_msg)) //, zmq::recv_flags::dontwait))
                {
                    _logger->debug("Got msg: {} ", _rcv_msg.to_string());
                    auto fgmonmsg = fg_monitor_message::from_string(_rcv_msg.to_string());
                    if (_upstream) {
                        fgmonmsg->set_schedid(_id);
                    }
                    if (fgmonmsg && _fgm) {
                        _fgm->push_message(fgmonmsg);
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            } catch (zmq::error_t err) {
                this->_rcv_done = true;
            }
        }
    });
    t.detach();
}

void fgm_proxy::push_message(fg_monitor_message_sptr msg)
{
    _logger->debug("Sending msg: {} ", msg->to_string());
    auto res = _client_socket.send(zmq::buffer(msg->to_string()), zmq::send_flags::none);
}

} // namespace gr