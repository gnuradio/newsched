#pragma once

#include <gnuradio/port.hpp>

namespace gr {

class node_interface
{

public:
    virtual void add_port(port_sptr p) = 0;
    virtual std::vector<port_sptr>& all_ports() = 0;
    virtual std::vector<port_sptr>& input_ports() = 0;
    virtual std::vector<port_sptr>& output_ports() = 0;
    virtual std::vector<port_sptr> input_stream_ports() = 0;
    virtual std::vector<port_sptr> output_stream_ports() = 0;
    virtual std::string& name() = 0;
    virtual std::string& alias() = 0;
    virtual uint32_t id() = 0;
    virtual void set_alias(std::string alias) = 0;
    virtual void set_id(uint32_t id) = 0;
    virtual port_sptr get_port(const std::string& name) = 0;
    virtual message_port_sptr get_message_port(const std::string& name) = 0;
    virtual port_sptr
    get_port(unsigned int index, port_type_t type, port_direction_t direction) = 0;
};


} // namespace gr
