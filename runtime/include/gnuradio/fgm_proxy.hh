#include <gnuradio/flowgraph_monitor.hh>

#include <gnuradio/scheduler.hh>
#include <zmq.hpp>
#include <thread>

namespace gr {


class fgm_proxy : public scheduler
{
private:
    flowgraph_monitor_sptr _fgm;
    bool _upstream;

    zmq::context_t _context;
    zmq::socket_t _server_socket;
    zmq::socket_t _client_socket;

    zmq::message_t _rcv_msg;

public:
    fgm_proxy(flowgraph_monitor_sptr fgm,
              const std::string& ipaddr,
              int port,
              bool upstream)
        : scheduler("fgm_proxy"),
          _fgm(fgm),
          _upstream(upstream),
          _context(1),
          _server_socket(_context, zmq::socket_type::pull),
          _client_socket(_context, zmq::socket_type::push)

    {
        int sndhwm = 1;
        int rcvhwm = 1;
        _server_socket.setsockopt(ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
        _server_socket.setsockopt(ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));

        _client_socket.setsockopt(ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
        _client_socket.setsockopt(ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));

        std::string svrendpoint = "tcp://*:" + std::to_string(upstream ? port : port + 1);
        _server_socket.bind(svrendpoint);

        std::string cliendpoint =
            "tcp://" + ipaddr + ":" + std::to_string(upstream ? port + 1 : port);
        _server_socket.connect(cliendpoint);

        std::thread t([this]() {
            while (true) { // (!this->_recv_done) {
                _rcv_msg.rebuild();
                auto res = _server_socket.recv(_rcv_msg, zmq::recv_flags::none);

                auto rcvstring = _rcv_msg.to_string();

                if (rcvstring == "DONE") {
                    auto fgmmsg = fg_monitor_message(fg_monitor_message_t::DONE, id());
                    _fgm->push_message(fgmmsg);
                } else if (rcvstring == "FLUSHED") {
                    auto fgmmsg = fg_monitor_message(fg_monitor_message_t::FLUSHED, id());
                    _fgm->push_message(fgmmsg);
                } else if (rcvstring == "KILL") {
                    auto fgmmsg = fg_monitor_message(fg_monitor_message_t::KILL, id());
                    _fgm->push_message(fgmmsg);
                }
            }
        });
        t.detach();
    }

    void push_message(scheduler_message_sptr msg)
    {
        if (msg->type() != scheduler_message_t::SCHEDULER_ACTION) {
            throw std::runtime_error(
                "Can only send scheduler action through fgm proxy interface");
        }

        auto action = std::static_pointer_cast<scheduler_action>(msg)->action();

        std::string outgoing = "";
        switch (action) {
        case scheduler_action_t::DONE:
            outgoing = "DONE";
        case scheduler_action_t::EXIT:
            outgoing = "EXIT";
        default:;
        }


        auto res = _client_socket.send(zmq::buffer(outgoing), zmq::send_flags::none);
    }
}; // namespace gr

} // namespace gr