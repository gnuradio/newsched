/* -*- c++ -*- */
/*
 * Copyright 2020 Josh Morman
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/domain_adapter.hpp>

namespace gr {


void domain_adapter_zmq_rep_svr::run_thread(sptr top) // zmq::socket_t* sock)
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
            zmq::message_t msg(0);
            sock->send(msg);
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
        case da_request_t::READ_INFO: {
            buffer_info_t info = top->buffer()->read_info();

            zmq::message_t msg(sizeof(buffer_info_t));
            memcpy(msg.data(), &info, sizeof(buffer_info_t));
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


void domain_adapter_zmq_req_cli::test()
{
    zmq::message_t response(100);
    std::string str_msg("hello");
    zmq::message_t msg(str_msg.begin(), str_msg.end());

    d_socket->send(msg);
    d_socket->recv(&response);

    std::string str = std::string(static_cast<char*>(response.data()), response.size());
    std::cout << str << std::endl;
}


buffer_info_t domain_adapter_zmq_req_cli::read_info()
{
    zmq::message_t msg(4);
    auto action = da_request_t::READ_INFO;
    memcpy(msg.data(), &action, 4);

    zmq::message_t response(sizeof(buffer_info_t));
    d_socket->send(msg);
    d_socket->recv(&response);

    buffer_info_t ret;
    memcpy(&ret, response.data(), sizeof(buffer_info_t));

    return ret;
}
buffer_info_t domain_adapter_zmq_req_cli::write_info()
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
void domain_adapter_zmq_req_cli::cancel()
{

    zmq::message_t msg(4);
    auto action = da_request_t::CANCEL;
    memcpy(msg.data(), &action, 4);

    zmq::message_t response(4);
    d_socket->send(msg);
    d_socket->recv(&response);
}

void domain_adapter_zmq_req_cli::post_read(int num_items)
{
    zmq::message_t msg(4 + sizeof(int));
    auto action = da_request_t::POST_READ;
    memcpy(msg.data(), &action, 4);
    memcpy((uint8_t*)msg.data() + 4, &num_items, sizeof(int));

    zmq::message_t response(0);
    d_socket->send(msg);
    d_socket->recv(&response);
}
void domain_adapter_zmq_req_cli::post_write(int num_items)
{
    zmq::message_t msg(4 + sizeof(int));
    auto action = da_request_t::POST_WRITE;
    memcpy(msg.data(), &action, 4);
    memcpy((uint8_t*)msg.data() + 4, &num_items, sizeof(int));

    zmq::message_t response(0);
    d_socket->send(msg);
    d_socket->recv(&response);
}

} // namespace gr