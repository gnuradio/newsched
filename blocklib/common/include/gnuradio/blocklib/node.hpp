/**
 * @file node.hpp
 * @author Josh Morman
 * @brief Anything that can be connected together in a graph, including graphs
 * @version 0.1
 * @date 2020-06-17
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <algorithm>

#include <gnuradio/blocklib/io_signature.hpp>
#include <gnuradio/blocklib/port.hpp>

namespace gr {
class node
{
protected:
    std::string d_name;
    std::string d_alias;
    io_signature d_input_signature;
    io_signature d_output_signature;
    std::vector<port_sptr> d_all_ports;
    std::vector<port_sptr> d_input_ports;
    std::vector<port_sptr> d_output_ports;

    void add_port(port_sptr p)
    {
        if (p->direction() == port_direction_t::INPUT) {
            if (p->type() == port_type_t::STREAM)
                p->set_index(input_stream_ports().size());
            // TODO: do message ports have an index??

            d_input_ports.push_back(p);

            // update the input_signature
            if (p->type() == port_type_t::STREAM) {
                d_input_signature = io_signature(sizeof_input_stream_ports());
            }
        } else if (p->direction() == port_direction_t::OUTPUT) {
            d_output_ports.push_back(p);

            if (p->type() == port_type_t::STREAM)
                p->set_index(output_stream_ports().size());

            if (p->type() == port_type_t::STREAM) {
                d_output_signature = io_signature(sizeof_output_stream_ports());
            }
        }
    }
    void remove_port(const std::string& name) {};  /// since ports are only added in constructor, is this necessary



public:
    node() : d_name("") {}
    node(const std::string& name) : d_name(name) {}
    virtual ~node() {}
    typedef std::shared_ptr<node> sptr;

    io_signature& input_signature() { return d_input_signature; };
    io_signature& output_signature() { return d_output_signature; };

    std::vector<port_sptr>& all_ports() { return d_all_ports; }
    std::vector<port_sptr>& input_ports() { return d_input_ports; }
    std::vector<port_sptr>& output_ports() { return d_output_ports; }
    std::vector<port_sptr> input_stream_ports()
    {
        std::vector<port_sptr> result;
        for (auto& p : d_input_ports)
            if (p->type() == port_type_t::STREAM)
                result.push_back(p);

        return result;
    }
    std::vector<port_sptr> output_stream_ports()
    {
        std::vector<port_sptr> result;
        for (auto& p : d_output_ports)
            if (p->type() == port_type_t::STREAM)
                result.push_back(p);

        return result;
    }

    std::vector<size_t> sizeof_input_stream_ports()
    {
        std::vector<size_t> result;
        for (auto& p : d_input_ports)
            if (p->type() == port_type_t::STREAM)
                result.push_back(p->data_size());

        return result;
    }

    std::vector<size_t> sizeof_output_stream_ports()
    {
        std::vector<size_t> result;
        for (auto& p : d_output_ports)
            if (p->type() == port_type_t::STREAM)
                result.push_back(p->data_size());

        return result;
    }

    std::string& name() { return d_name; };
    std::string& alias() { return d_alias; }
    void set_alias(std::string alias) { d_alias = alias; }

    port_sptr get_port(std::string& name, port_type_t type, port_direction_t direction)
    {
        auto pred = [name, type, direction](port_sptr p) {
            return (p->type() == type && p->direction() == direction &&
                    p->name() == name);
        };
        std::vector<port_sptr>::iterator it =
            std::find_if(std::begin(d_all_ports), std::end(d_all_ports), pred);

        if (it != std::end(d_all_ports)) {
            return *it;
        } else {
            // port was not found
            return nullptr;
        }
    }
    port_sptr get_port(unsigned int index, port_type_t type, port_direction_t direction)
    {
        auto pred = [index, type, direction](port_sptr p) {
            return (p->type() == type && p->direction() == direction &&
                    p->index() == index);
        };
        std::vector<port_sptr>::iterator it =
            std::find_if(std::begin(d_all_ports), std::end(d_all_ports), pred);

        if (it != std::end(d_all_ports)) {
            return *it;
        } else {
            // port was not found
            return nullptr;
        }
    }
};

typedef node::sptr node_sptr;
typedef std::vector<node_sptr> node_vector_t;

} // namespace gr