#include <gnuradio/domain_adapter_zmq.hpp>

namespace gr {


domain_adapter_zmq_rep_svr::domain_adapter_zmq_rep_svr(const std::string& endpoint_uri)
    : domain_adapter(buffer_location_t::LOCAL)
{
    d_context = new zmq::context_t(1);
    d_socket = new zmq::socket_t(*d_context,
                                 zmq::socket_type::rep); // server is a rep socket
    int time = 0;
    int hwm = 64;
    d_socket->setsockopt(ZMQ_LINGER, &time, sizeof(time));
    d_socket->setsockopt(ZMQ_SNDHWM, &hwm, sizeof(int));
    d_socket->setsockopt(ZMQ_RCVHWM, &hwm, sizeof(int));
    d_socket->bind(endpoint_uri);
}

void domain_adapter_zmq_rep_svr::start_thread(sptr ptr)
{
    d_thread = std::thread(run_thread, ptr);
}

void domain_adapter_zmq_rep_svr::run_thread(sptr top) // zmq::socket_t* sock)
{
    auto sock = top->d_socket;
    while (1) {
        zmq::message_t request(100);
        sock->recv(&request);

        // Parse the message
        auto action = *((da_request_t*)request.data());
        switch (action) {
        case da_request_t::WRITE_INFO: {
            buffer_info_t info;
            zmq::message_t msg(4 + sizeof(buffer_info_t));
            if (top->buffer()->write_info(info)) {
                auto action = da_response_t::OK;
                memcpy(msg.data(), &action, 4);
                memcpy((uint8_t*)msg.data() + 4, &info, sizeof(buffer_info_t));
            } else {
                auto action = da_response_t::ERROR;
            }

            sock->send(msg);
        } break;
        case da_request_t::POST_WRITE: {
            int num_items;
            memcpy(&num_items, (uint8_t*)request.data() + 4, sizeof(int));
            top->buffer()->post_write(num_items);


            zmq::message_t msg(0);
            sock->send(msg);
        } break;
        case da_request_t::READ_INFO: {
            buffer_info_t info;
            zmq::message_t msg(4 + sizeof(buffer_info_t));
            if (top->buffer()->read_info(info)) {
                auto action = da_response_t::OK;
                memcpy(msg.data(), &action, 4);
                memcpy((uint8_t*)msg.data() + 4, &info, sizeof(buffer_info_t));
            } else {
                auto action = da_response_t::ERROR;
            }

            sock->send(msg);
        } break;
        case da_request_t::POST_READ: {
            int num_items;
            memcpy(&num_items, (uint8_t*)request.data() + 4, sizeof(int));
            top->buffer()->post_read(num_items);

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


domain_adapter_zmq_req_cli::domain_adapter_zmq_req_cli(const std::string& endpoint_uri)
    : domain_adapter(buffer_location_t::REMOTE)
{
    d_context = new zmq::context_t(1);
    d_socket = new zmq::socket_t(*d_context,
                                 zmq::socket_type::req); // server is a rep socket
    int time = 0;
    int hwm = 64;
    d_socket->setsockopt(ZMQ_LINGER, &time, sizeof(time));
    d_socket->setsockopt(ZMQ_SNDHWM, &hwm, sizeof(int));
    d_socket->setsockopt(ZMQ_RCVHWM, &hwm, sizeof(int));
    d_socket->connect(endpoint_uri);
}


bool domain_adapter_zmq_req_cli::read_info(buffer_info_t& info)
{
    zmq::message_t msg(4);
    auto action = da_request_t::READ_INFO;
    memcpy(msg.data(), &action, 4);

    zmq::message_t response(4 + sizeof(buffer_info_t));
    d_socket->send(msg);
    d_socket->recv(&response);
    auto res_code = *((da_response_t*)response.data());
    if (res_code == da_response_t::OK) {
        memcpy(&info, (uint8_t*)response.data() + 4, sizeof(buffer_info_t));
        return true;
    } else {
        return false;
    }
}
bool domain_adapter_zmq_req_cli::write_info(buffer_info_t& info)
{
    zmq::message_t msg(4);
    auto action = da_request_t::WRITE_INFO;
    memcpy(msg.data(), &action, 4);

    zmq::message_t response(4 + sizeof(buffer_info_t));
    d_socket->send(msg);
    d_socket->recv(&response);

    auto res_code = *((da_response_t*)response.data());
    if (res_code == da_response_t::OK) {
        memcpy(&info, (uint8_t*)response.data() + 4, sizeof(buffer_info_t));
        return true;
    } else {
        return false;
    }
}

void domain_adapter_zmq_req_cli::post_read(int num_items)
{
    zmq::message_t msg(4 + sizeof(int));
    auto action = da_request_t::POST_READ;
    memcpy(msg.data(), &action, 4);
    memcpy((uint8_t*)msg.data() + 4, &num_items, sizeof(int));

    zmq::message_t response(4);
    d_socket->send(msg);
    d_socket->recv(&response);
}
void domain_adapter_zmq_req_cli::post_write(int num_items)
{
    zmq::message_t msg(4 + sizeof(int));
    auto action = da_request_t::POST_WRITE;
    memcpy(msg.data(), &action, 4);
    memcpy((uint8_t*)msg.data() + 4, &num_items, sizeof(int));

    zmq::message_t response(4);
    d_socket->send(msg);
    d_socket->recv(&response);
}


} // namespace gr