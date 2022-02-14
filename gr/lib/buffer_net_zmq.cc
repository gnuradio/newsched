#include <gnuradio/buffer_cpu_vmcirc.hh>
#include <gnuradio/buffer_net_zmq.hh>
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>
namespace gr {


std::shared_ptr<buffer_properties>
buffer_net_zmq_properties::make_from_params(const std::string& json_str)
{
    auto json_obj = nlohmann::json::parse(json_str);
    return make(json_obj["ipaddr"], json_obj["port"]);
}

buffer_sptr buffer_net_zmq::make(size_t num_items,
                                 size_t item_size,
                                 std::shared_ptr<buffer_properties> buffer_properties)
{

    auto zbp = std::static_pointer_cast<buffer_net_zmq_properties>(buffer_properties);
    if (zbp != nullptr) {
        return buffer_sptr(
            new buffer_net_zmq(num_items, item_size, buffer_properties, zbp->port()));
    } else {
        throw std::runtime_error(
            "Failed to cast buffer properties to buffer_net_zmq_properties");
    }
}

buffer_net_zmq::buffer_net_zmq(size_t num_items,
                               size_t item_size,
                               std::shared_ptr<buffer_properties> buf_properties,
                               int port)
    : buffer(num_items, item_size, buf_properties),
      _context(1),
      _socket(_context, zmq::socket_type::push)
{
    _debug_logger = logging::get_logger("buffer_net_zmq", "debug");
    set_type("buffer_net_zmq");
    _buffer.resize(_buf_size);
    int sndhwm = 1;
    int rcvhwm = 1;
    _socket.setsockopt(ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
    _socket.setsockopt(ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));
    std::string endpoint = "tcp://*:" + std::to_string(port);
    std::cout << "snd_endpoint: " << endpoint << std::endl;
    _socket.bind(endpoint);
}


/****************************************************************************/
/*   READER METHODS                                                         */
/****************************************************************************/


buffer_reader_sptr
buffer_net_zmq_reader::make(size_t itemsize, std::shared_ptr<buffer_properties> buf_props)
{
    auto zbp = std::static_pointer_cast<buffer_net_zmq_properties>(buf_props);
    if (zbp != nullptr) {
        return buffer_reader_sptr(
            new buffer_net_zmq_reader(buf_props, itemsize, zbp->ipaddr(), zbp->port()));
    } else {
        throw std::runtime_error(
            "Failed to cast buffer properties to buffer_net_zmq_properties");
    }
}

buffer_net_zmq_reader::buffer_net_zmq_reader(std::shared_ptr<buffer_properties> buf_props,
                                             size_t itemsize,
                                             const std::string& ipaddr,
                                             int port)
    : buffer_reader(nullptr, buf_props, itemsize, 0),
      _context(1),
      _socket(_context, zmq::socket_type::pull)
{
    _debug_logger = logging::get_logger("buffer_net_zmq_reader", "debug");
    auto bufprops = std::make_shared<buffer_cpu_vmcirc_properties>();
    _circbuf = gr::buffer_cpu_vmcirc::make(
        8192,
        itemsize,
        bufprops); // FIXME - make nitems a buffer reader factory parameter
    _circbuf_rdr = _circbuf->add_reader(bufprops, itemsize);

    // auto b = (float *)_circbuf->write_ptr();
    
    // for (int i=0; i<8192; i++)
    // {
    //     b[i] = i;
    // }
    // _circbuf->post_write(8192);

    // buffer_info_t info;
    // _circbuf_rdr->read_info(info); 
    // auto br = (float *)_circbuf_rdr->read_ptr();
    

    // _circbuf_rdr->post_read(4096);
    // br = (float *) _circbuf_rdr->read_ptr();
    

    std::string endpoint = "tcp://" + ipaddr + ":" + std::to_string(port);
    GR_LOG_DEBUG(_debug_logger, "rcv_endpoint: {}", endpoint);
    int sndhwm = 1;
    int rcvhwm = 1;
    _socket.setsockopt(ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
    _socket.setsockopt(ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));
    // _socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);
    _socket.connect(endpoint);
    GR_LOG_DEBUG(_debug_logger, "   ... connected");

    std::thread t([this]() {
        while (!this->_recv_done) {
            // zmq::message_t msg{};
            // See how much room we have in the circular buffer
            buffer_info_t wi;
            _circbuf->write_info(wi);

            auto n_bytes_left_in_msg = _msg.size() - _msg_idx;
            auto n_bytes_in_circbuf = wi.n_items * wi.item_size;
            auto bytes_to_write = std::min(n_bytes_in_circbuf, n_bytes_left_in_msg);
            auto items_to_write = bytes_to_write / wi.item_size;
            bytes_to_write = items_to_write * wi.item_size;

            if (n_bytes_in_circbuf <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            if (bytes_to_write > 0)
            {
                memcpy(wi.ptr, (uint8_t *)_msg.data() + _msg_idx, bytes_to_write);
                GR_LOG_DEBUG(_debug_logger, "copied {} items", bytes_to_write / wi.item_size);
                _msg_idx += bytes_to_write;
                n_bytes_left_in_msg = _msg.size() - _msg_idx;
                _circbuf->post_write(items_to_write);
                notify_scheduler();
            }

            if (n_bytes_left_in_msg == 0)
            {
                _msg.rebuild();
                GR_LOG_DEBUG(_debug_logger, "going into recv");
                auto r = _socket.recv(_msg, zmq::recv_flags::none);
                if (r) {
                    GR_LOG_DEBUG(_debug_logger, "received msg with size {} items", _msg.size() / wi.item_size);
                    _msg_idx = 0;
                }
            }
            // GR_LOG_DEBUG(_debug_logger, "recv: {}", wi.n_items);
            // auto ret = this->_socket.recv(
            //     zmq::mutable_buffer(_circbuf->write_ptr(), wi.n_items * wi.item_size),
            //     zmq::recv_flags::none);

            // _circbuf->post_write(wi.n_items);
            // notify_scheduler();
            // auto recbuf = *ret;
            // assert(recbuf.size == wi.n_items * wi.item_size);

            // GR_LOG_DEBUG(_debug_logger, "nbytesrcv: {}", recbuf.size);
            // std::cout << "    ---> msg received " << msg.size() << " bytes" <<
            // std::endl;
        }
    });

    t.detach();
}

} // namespace gr
