#pragma once

#include <zmq.hpp>
#include <thread>
#include <gnuradio/logging.hh>

namespace gr {

class flowgraph_monitor;
typedef std::shared_ptr<flowgraph_monitor> flowgraph_monitor_sptr;

class fg_monitor_message;
typedef std::shared_ptr<fg_monitor_message> fg_monitor_message_sptr;
class fgm_proxy
{
private:
    flowgraph_monitor_sptr _fgm;
    bool _upstream;

    zmq::context_t _context;
    zmq::socket_t _server_socket;
    zmq::socket_t _client_socket;

    zmq::message_t _rcv_msg;

    int _id;

    logger_sptr _logger;

    bool _rcv_done = false;

public:
    typedef std::shared_ptr<fgm_proxy> sptr;
    static sptr make(const std::string& ipaddr, int port, bool upstream);
    fgm_proxy(const std::string& ipaddr, int port, bool upstream);
    void push_message(fg_monitor_message_sptr msg);

    void set_fgm(flowgraph_monitor_sptr fgm) { _fgm = fgm; }
    int id() { return _id; }
    void set_id(int id_) { _id = id_; }
    bool upstream() { return _upstream; }

    void kill() { _rcv_done = true;
    _context.shutdown();
    _client_socket.close();
    _server_socket.close();
    _context.close(); }
}; // namespace gr
typedef fgm_proxy::sptr fgm_proxy_sptr;


} // namespace gr