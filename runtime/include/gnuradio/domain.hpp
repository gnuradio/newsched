#pragma once

#include <iostream>
#include <memory>

#include <gnuradio/block.hpp>
#include <gnuradio/domain_adapter.hpp>
#include <gnuradio/graph.hpp>
#include <gnuradio/scheduler.hpp>

namespace gr {


class domain_conf
{
public:
    domain_conf(scheduler_sptr sched,
                std::vector<block_sptr> blocks,
                domain_adapter_conf_sptr da_conf = nullptr, // nullptr assumes default domain adapter conf
                domain_adapter_conf_per_edge da_edge_confs =
                    domain_adapter_conf_per_edge())
        : _sched(sched), _blocks(blocks), _da_conf(da_conf), _da_edge_confs(da_edge_confs)
    {
    }

    auto sched() { return _sched; }
    auto blocks() { return _blocks; }
    auto da_conf() { return _da_conf; }
    auto da_edge_confs() { return _da_edge_confs; }

private:
    scheduler_sptr _sched;
    std::vector<block_sptr> _blocks;
    domain_adapter_conf_sptr _da_conf;
    domain_adapter_conf_per_edge _da_edge_confs;
};

typedef std::vector<domain_conf> domain_conf_vec;

} // namespace gr
