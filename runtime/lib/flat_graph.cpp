/* -*- c++ -*- */
/*
 * Copyright 2007,2011,2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <assert.h>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <gnuradio/flat_graph.hpp>

namespace gr {

#define FLOWGRAPH_DEBUG 0

// edge::~edge() {}

flat_graph::flat_graph() {}

flat_graph::~flat_graph() {}


// void flat_graph::validate()
// {
//     d_blocks = calc_used_blocks();

//     for (auto &b : d_blocks)
//     {
//         std::vector<int> used_ports;
//         int ninputs, noutputs;

//         if (FLOWGRAPH_DEBUG)
//             std::cout << "Validating block: " << b << std::endl;

//         used_ports = calc_used_ports(b, true); // inputs
//         ninputs = used_ports.size();
//         check_contiguity(b, used_ports, true); // inputs

//         used_ports = calc_used_ports(b, false); // outputs
//         noutputs = used_ports.size();
//         check_contiguity(b, used_ports, false); // outputs

//         // if (!((*p)->check_topology(ninputs, noutputs))) {
//         //     std::stringstream msg;
//         //     msg << "check topology failed on " << (*p) << " using ninputs=" <<
//         ninputs
//         //         << ", noutputs=" << noutputs;
//         //     throw std::runtime_error(msg.str());
//         // }

//         // update the block alias
//         std::map<std::string, int> name_count;
//         // look in the map, see how many of that name exist
//         // make the alias name + count;
//         // increment the map
//         int cnt;
//         if (name_count.find(b->name()) == name_count.end()) {
//             name_count[b->name()] = cnt = 0;
//         } else {
//             cnt = name_count[b->name()];
//         }
//         b->set_alias(b->name() + std::to_string(cnt));
//         name_count[b->name()] = cnt+1;


//     }
// }

void flat_graph::clear()
{
    // Boost shared pointers will deallocate as needed
    d_blocks.clear();

    graph::clear();
}

void flat_graph::check_valid_port(block_sptr block, port_sptr port)
{
    // std::stringstream msg;

    // if (port < 0) {
    //     msg << "negative port number " << port << " is invalid";
    //     throw std::invalid_argument(msg.str());
    // }

    // if (port >= sig.n_streams()) {
    //     msg << "port number " << port << " exceeds max of " << sig.n_streams() - 1;
    //     throw std::invalid_argument(msg.str());
    // }
}

void flat_graph::check_dst_not_used(const block_endpoint& dst)
{
    // A destination is in use if it is already on the edge list
    for (edge_viter_t p = _edges.begin(); p != _edges.end(); p++)
        if (p->dst() == dst) {
            std::stringstream msg;
            msg << "destination already in use by edge " << (*p);
            throw std::invalid_argument(msg.str());
        }
}

void flat_graph::check_type_match(const block_endpoint& src, const block_endpoint& dst)
{
    int src_size = src.port()->data_size();
    int dst_size = dst.port()->data_size();

    // TODO: enforce strongly typed checking??

    if (src_size != dst_size) {
        std::stringstream msg;
        msg << "itemsize mismatch: " << src << " using " << src_size << ", " << dst
            << " using " << dst_size;
        throw std::invalid_argument(msg.str());
    }
}

block_vector_t flat_graph::calc_used_blocks()
{
    block_vector_t tmp;

    // Collect all blocks in the edge list
    for (edge_viter_t p = _edges.begin(); p != _edges.end(); p++) {
        tmp.push_back(static_cast<block_endpoint>(p->src()).block());
        tmp.push_back(static_cast<block_endpoint>(p->dst()).block());
    }

    return unique_vector<block_sptr>(tmp);
}

port_vector_t flat_graph::calc_used_ports(block_sptr block, bool check_inputs)
{
    port_vector_t tmp;

    // Collect all seen ports
    edge_vector_t edges = calc_connections(block, check_inputs);
    for (edge_viter_t p = edges.begin(); p != edges.end(); p++) {
        if (check_inputs == true)
            tmp.push_back(p->dst().port());
        else
            tmp.push_back(p->src().port());
    }

    return unique_vector<port_sptr>(tmp);
}

edge_vector_t flat_graph::calc_connections(block_sptr block, bool check_inputs)
{
    edge_vector_t result;

    for (edge_viter_t p = _edges.begin(); p != _edges.end(); p++) {
        if (check_inputs) {
            if (static_cast<block_endpoint>(p->dst()).block() == block)
                result.push_back(*p);
        } else {
            if (static_cast<block_endpoint>(p->src()).block() == block)
                result.push_back(*p);
        }
    }

    return result; // assumes no duplicates
}

void flat_graph::check_contiguity(block_sptr block,
                                  const port_vector_t used_ports,
                                  bool check_inputs)
{

    // FIXME: make this function do the following:
    //  look at the block and all its ports
    //  make sure all the ports that are not optional are in the list of used ports

    // std::stringstream msg;

    // gr::io_signature sig =
    //     check_inputs ? block->input_signature() : block->output_signature();

    // int nports = used_ports.size();

    // // TODO - make io_signature deal with optional ports -- port class??

    // int min_ports = sig.n_streams();
    // int max_ports = sig.n_streams();

    // if (nports == 0 && min_ports == 0)
    //     return;

    // if (nports < min_ports) {
    //     msg << block << ": insufficient connected "
    //         << (check_inputs ? "input ports " : "output ports ") << "(" << min_ports
    //         << " needed, " << nports << " connected)";
    //     throw std::runtime_error(msg.str());
    // }

    // if (nports > max_ports && max_ports != io_signature::IO_INFINITE) {
    //     msg << block << ": too many connected "
    //         << (check_inputs ? "input ports " : "output ports ") << "(" << max_ports
    //         << " allowed, " << nports << " connected)";
    //     throw std::runtime_error(msg.str());
    // }

    // if (used_ports[nports - 1] + 1 != nports) {
    //     for (int i = 0; i < nports; i++) {
    //         if (used_ports[i] != i) {
    //             msg << block << ": missing connection "
    //                 << (check_inputs ? "to input port " : "from output port ") << i;
    //             throw std::runtime_error(msg.str());
    //         }
    //     }
    // }
}

block_vector_t flat_graph::calc_downstream_blocks(block_sptr block, port_sptr port)
{
    block_vector_t tmp;

    for (edge_viter_t p = _edges.begin(); p != _edges.end(); p++)
        if (static_cast<block_endpoint>(p->src()) == block_endpoint(block, port))
            tmp.push_back(static_cast<block_endpoint>(p->dst()).block());

    return unique_vector<block_sptr>(tmp);
}

block_vector_t flat_graph::calc_downstream_blocks(block_sptr block)
{
    block_vector_t tmp;

    for (edge_viter_t p = _edges.begin(); p != _edges.end(); p++)
        if (static_cast<block_endpoint>(p->src()).block() == block)
            tmp.push_back(static_cast<block_endpoint>(p->dst()).block());

    return unique_vector<block_sptr>(tmp);
}

edge_vector_t flat_graph::calc_upstream_edges(block_sptr block)
{
    edge_vector_t result;

    for (edge_viter_t p = _edges.begin(); p != _edges.end(); p++)
        if (static_cast<block_endpoint>(p->dst()).block() == block)
            result.push_back(*p);

    return result; // Assume no duplicates
}

bool flat_graph::has_block_p(block_sptr block)
{
    block_viter_t result;
    result = std::find(d_blocks.begin(), d_blocks.end(), block);
    return (result != d_blocks.end());
}


std::vector<block_vector_t> flat_graph::partition()
{
    std::vector<block_vector_t> result;
    block_vector_t blocks = calc_used_blocks();
    block_vector_t graph;

    while (!blocks.empty()) {
        graph = calc_reachable_blocks(blocks[0], blocks);
        assert(graph.size());
        result.push_back(topological_sort(graph));

        for (block_viter_t p = graph.begin(); p != graph.end(); p++)
            blocks.erase(find(blocks.begin(), blocks.end(), *p));
    }

    return result;
}

block_vector_t flat_graph::calc_reachable_blocks(block_sptr block, block_vector_t& blocks)
{
    block_vector_t result;

    // Mark all blocks as unvisited
    for (block_viter_t p = blocks.begin(); p != blocks.end(); p++)
        (*p)->set_color(block::WHITE);

    // Recursively mark all reachable blocks
    reachable_dfs_visit(block, blocks);

    // Collect all the blocks that have been visited
    for (block_viter_t p = blocks.begin(); p != blocks.end(); p++)
        if ((*p)->color() == block::BLACK)
            result.push_back(*p);

    return result;
}

// Recursively mark all reachable blocks from given block and block list
void flat_graph::reachable_dfs_visit(block_sptr block, block_vector_t& blocks)
{
    // Mark the current one as visited
    block->set_color(block::BLACK);

    // Recurse into adjacent vertices
    block_vector_t adjacent = calc_adjacent_blocks(block, blocks);

    for (block_viter_t p = adjacent.begin(); p != adjacent.end(); p++)
        if ((*p)->color() == block::WHITE)
            reachable_dfs_visit(*p, blocks);
}

// Return a list of block adjacent to a given block along any edge
block_vector_t flat_graph::calc_adjacent_blocks(block_sptr block, block_vector_t& blocks)
{
    block_vector_t tmp;

    // Find any blocks that are inputs or outputs
    for (edge_viter_t p = _edges.begin(); p != _edges.end(); p++) {
        if (p->src().node() == block)
            tmp.push_back(static_cast<block_endpoint>(p->dst()).block());
        if (p->dst().node() == block)
            tmp.push_back(static_cast<block_endpoint>(p->src()).block());
    }

    return unique_vector<block_sptr>(tmp);
}

block_vector_t flat_graph::topological_sort(block_vector_t& blocks)
{
    block_vector_t tmp;
    block_vector_t result;
    tmp = sort_sources_first(blocks);

    // Start 'em all white
    for (block_viter_t p = tmp.begin(); p != tmp.end(); p++)
        (*p)->set_color(block::WHITE);

    for (block_viter_t p = tmp.begin(); p != tmp.end(); p++) {
        if ((*p)->color() == block::WHITE)
            topological_dfs_visit(*p, result);
    }

    reverse(result.begin(), result.end());
    return result;
}

block_vector_t flat_graph::sort_sources_first(block_vector_t& blocks)
{
    block_vector_t sources, nonsources, result;

    for (block_viter_t p = blocks.begin(); p != blocks.end(); p++) {
        if (source_p(*p))
            sources.push_back(*p);
        else
            nonsources.push_back(*p);
    }

    for (block_viter_t p = sources.begin(); p != sources.end(); p++)
        result.push_back(*p);

    for (block_viter_t p = nonsources.begin(); p != nonsources.end(); p++)
        result.push_back(*p);

    return result;
}

bool flat_graph::source_p(block_sptr block) { return calc_upstream_edges(block).empty(); }

void flat_graph::topological_dfs_visit(block_sptr block, block_vector_t& output)
{
    block->set_color(block::GREY);
    block_vector_t blocks(calc_downstream_blocks(block));

    for (block_viter_t p = blocks.begin(); p != blocks.end(); p++) {
        switch ((*p)->color()) {
        case block::WHITE:
            topological_dfs_visit(*p, output);
            break;

        case block::GREY:
            throw std::runtime_error("flow graph has loops!");

        case block::BLACK:
            continue;

        default:
            throw std::runtime_error("invalid color on block!");
        }
    }


    output.push_back(block);
}

typedef std::shared_ptr<flat_graph> flat_graph_sptr;

} /* namespace gr */
