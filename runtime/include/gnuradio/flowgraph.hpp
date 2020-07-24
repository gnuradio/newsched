/* -*- c++ -*- */
/*
 * Copyright 2006,2007,2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_GR_RUNTIME_FLOWGRAPH_H
#define INCLUDED_GR_RUNTIME_FLOWGRAPH_H

#include "api.h"
#include <iostream>
#include <memory>

#include <gnuradio/blocklib/block.hpp>
#include <gnuradio/domain_adapter.hpp>
#include <gnuradio/graph.hpp>
#include <gnuradio/scheduler.hpp>
#include <gnuradio/domain.hpp>

namespace gr {

// typedef std::tuple<scheduler_sptr, std::vector<block_sptr>> partition_conf;
// typedef std::vector<partition_conf> partition_conf_vec;

class flowgraph : public graph
{
private:
    std::vector<scheduler_sptr> d_schedulers;
    flat_graph_sptr d_flat_graph;
    std::vector<graph_sptr> d_subgraphs;
    std::vector<flat_graph_sptr> d_flat_subgraphs;

public:
    flowgraph(){};
    typedef std::shared_ptr<flowgraph> sptr;
    virtual ~flowgraph(){};
    void set_scheduler(scheduler_sptr sched)
    {
        d_schedulers = std::vector<scheduler_sptr>{ sched };
    }
    void set_schedulers(std::vector<scheduler_sptr> sched) { d_schedulers = sched; }
    void add_scheduler(scheduler_sptr sched) { d_schedulers.push_back(sched); }
    void clear_schedulers() { d_schedulers.clear(); }
    void partition(std::vector<domain_conf>& confs)
    {
        // Create new subgraphs based on the partition configuration
        d_subgraphs.clear();

        // std::vector<std::tuple<graph,edge>> domain_crossings;
        std::vector<edge> domain_crossings;
        std::vector<domain_conf> crossing_confs;
        std::vector<scheduler_sptr> partition_scheds;

        for (auto& conf : confs) {
            auto g = graph::make(); // create a new subgraph
            // Go through the blocks assigned to this scheduler
            // See whether they connect to the same graph or account for a domain crossing

            auto sched = conf.sched(); //std::get<0>(conf);
            auto blocks = conf.blocks(); //std::get<1>(conf);
            for (auto b : blocks) // for each of the blocks in the tuple
            {
                for (auto input_port : b->input_stream_ports()) {
                    auto edges = find_edge(input_port);
                    // There should only be one edge connected to an input port
                    // Crossings associated with the downstream port
                    auto e = edges[0];
                    auto other_block = e.src().node();

                    // Is the other block in our current partition
                    if (std::find(blocks.begin(), blocks.end(), other_block) !=
                        blocks.end()) {
                        g->connect(e.src(), e.dst());
                    } else {
                        // add this edge to the list of domain crossings
                        // domain_crossings.push_back(std::make_tuple(g,e));
                        domain_crossings.push_back(e);
                        crossing_confs.push_back(conf);
                    }
                }
            }

            d_subgraphs.push_back(g);
            partition_scheds.push_back(sched);
        }

        // Now, let's set up domain adapters at the domain crossings
        // Several assumptions are being made now:
        //   1.  All schedulers running on the same processor
        //   2.  Outputs that cross domains can only be mapped one input
        //   3.  Fixed client/server relationship - limited configuration of DA

        int crossing_index = 0;
        for (auto c : domain_crossings) {
            // Attach a domain adapter to the src and dst ports of the edge
            // auto g = std::get<0>(c);
            // auto e = std::get<1>(c);

            // Find the subgraph that holds src block
            graph_sptr src_block_graph = nullptr;
            for (auto g : d_subgraphs) {
                auto blocks = g->calc_used_nodes();
                if (std::find(blocks.begin(), blocks.end(), c.src().node()) !=
                    blocks.end()) {
                    src_block_graph = g;
                    break;
                }
            }

            // Find the subgraph that holds dst block
            graph_sptr dst_block_graph = nullptr;
            for (auto g : d_subgraphs) {
                auto blocks = g->calc_used_nodes();
                if (std::find(blocks.begin(), blocks.end(), c.dst().node()) !=
                    blocks.end()) {
                    dst_block_graph = g;
                    break;
                }
            }

            if (!src_block_graph || !dst_block_graph) {
                throw std::runtime_error("Cannot find both sides of domain adapter");
            }

            // Create Domain Adapter pair
            // right now, only one port - have a list of available ports
            // put the buffer downstream
            auto conf = crossing_confs[crossing_index];
            
            // Does the crossing have a specific domain adapter defined
            domain_adapter_conf_sptr da_conf = nullptr;
            for (auto ec : conf.da_edge_confs())
            {
                auto conf_edge = std::get<0>(ec);
                if (c == conf_edge)
                {
                    da_conf = std::get<1>(ec);
                    break;
                }
            }


            // else if defined: use the default defined for the domain
            if (!da_conf)
            {
                da_conf = conf.da_conf();
            }

            // else, use the default domain adapter configuration ??
            // TODO


#if 0
            auto da_src =
                domain_adapter_zmq_req_cli::make(std::string("tcp://127.0.0.1:1234"),
                                                 buffer_preference_t::UPSTREAM,
                                                 c.src().port());
            auto da_dst =
                domain_adapter_zmq_rep_svr::make(std::string("tcp://127.0.0.1:1234"),
                                                 buffer_preference_t::UPSTREAM,
                                                 c.dst().port());
#endif
            // use the conf to produce the domain adapters

            auto da_pair = da_conf->make_domain_adapter_pair(c.src().port(), c.dst().port());
            auto da_src = std::get<0>(da_pair);
            auto da_dst = std::get<1>(da_pair);


            // da_src->test();

            // Attach domain adapters to the src and dest blocks
            // domain adapters only have one port
            src_block_graph->connect(c.src(),
                                     node_endpoint(da_src, da_src->all_ports()[0]));
            dst_block_graph->connect(node_endpoint(da_dst, da_dst->all_ports()[0]),
                                     c.dst());


            crossing_index++;
        }

        d_flat_subgraphs.clear();
        for (auto i = 0; i < partition_scheds.size(); i++) {
            d_flat_subgraphs.push_back(flat_graph::make_flat(d_subgraphs[i]));
            partition_scheds[i]->initialize(d_flat_subgraphs[i]);
        }
    }
    void validate()
    {
        d_flat_graph = flat_graph::make_flat(base());
        for (auto sched : d_schedulers)
            sched->initialize(d_flat_graph);
    }
    void start()
    {
        for (auto s : d_schedulers) {
            s->start();
        }
    }
    void stop()
    {
        for (auto s : d_schedulers) {
            s->stop();
        }
    }
    void wait()
    {
        for (auto s : d_schedulers) {
            s->wait();
        }
    }
};

typedef flowgraph::sptr flowgraph_sptr;
} // namespace gr

#endif
