#pragma once

#include <algorithm>
#include <vector>

#include <gnuradio/logging.hpp>
#include <gnuradio/nodeid_generator.hpp>
#include <gnuradio/port.hpp>

namespace gr {

typedef uint32_t nodeid_t;

/**
 * @brief Base class for node
 *
 * The node class represents all things that can be connected together in graphs.  Nodes
 * should not be instantiated directly but through the derived classes (e.g. block or
 * graph)
 * 
 * Nodes hold ports, have a name, alias, and universal ID, but that's about it
 * 
 * Ports are added after construction of the node to simplify the constructor
 *
 */
class node
{
protected:
    std::string d_name;
    std::string d_alias;
    nodeid_t d_id; // Unique number given to block from runtime
    std::vector<port_sptr> d_all_ports;
    std::vector<port_sptr> d_input_ports;
    std::vector<port_sptr> d_output_ports;

    logger_sptr _logger;
    logger_sptr _debug_logger;

    void add_port(port_sptr p)
    {
        d_all_ports.push_back(p);
        if (p->direction() == port_direction_t::INPUT) {
            if (p->type() == port_type_t::STREAM)
                p->set_index(input_stream_ports().size());
            // TODO: do message ports have an index??

            d_input_ports.push_back(p);
        } else if (p->direction() == port_direction_t::OUTPUT) {
            if (p->type() == port_type_t::STREAM)
                p->set_index(output_stream_ports().size());

            d_output_ports.push_back(p);
        }
    }
    void remove_port(const std::string& name){}; /// since ports are only added in
                                                 /// constructor, is this necessary


public:
    node() : d_name("") {}
    node(const std::string& name) : d_name(name), d_alias(name) { d_id = nodeid_generator::get_id(); }
    virtual ~node() {}
    typedef std::shared_ptr<node> sptr;

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
    uint32_t id() { return d_id; }
    void set_alias(std::string alias)
    {
        d_alias = alias;

        // Instantiate the loggers when the alias is set
        _logger = logging::get_logger(alias, "default");
        _debug_logger = logging::get_logger(alias + "_dbg", "debug");
    }

    void set_id(uint32_t id) { d_id = id; }

    port_sptr get_port(const std::string& name)
    {
        auto pred = [name](port_sptr p) {
            return (p->name() == name);
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

    message_port_sptr get_message_port(const std::string& name)
    {
        auto p = get_port(name);

        // could be null if the requested port is not a message port
        return std::dynamic_pointer_cast<message_port>(p);
    }

    port_sptr get_port(unsigned int index, port_type_t type, port_direction_t direction)
    {
        auto pred = [index, type, direction](port_sptr p) {
            return (p->type() == type && p->direction() == direction &&
                    p->index() == (int)index);
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
