#pragma once

#include <iostream>
#include <memory>
#include <string>

#include <gnuradio/block.hh>
#include <gnuradio/graph.hh>
#include <gnuradio/scheduler.hh>

namespace gr {


class execution_host_properties
{
    // Later break this out into dynamic classes
    std::string _ipaddr;
    int _port;

    public:
    execution_host_properties(const std::string& ipaddr, int port) : 
    _ipaddr(ipaddr), _port(port)
    {

    }

    static std::shared_ptr<execution_host_properties> make(const std::string& ipaddr, int port)
    {
        return std::make_shared<execution_host_properties>(ipaddr, port);
    }
};
typedef std::shared_ptr<execution_host_properties> execution_host_properties_sptr;

/**
 * @brief Domain Configuration
 * 
 * A struct to contain all the necessary information about a domain:
 *  - Scheduler
 *  - Blocks
 *  
 */
class domain_conf
{
public:
    domain_conf(scheduler_sptr sched,
                std::vector<node_sptr> blocks,
                execution_host_properties_sptr host = nullptr)
        : _sched(sched), _blocks(blocks)
    {
    }

    auto sched() { return _sched; }
    auto blocks() { return _blocks; }

private:
    scheduler_sptr _sched;
    std::vector<node_sptr> _blocks;

};

typedef std::vector<domain_conf> domain_conf_vec;

} // namespace gr
