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

#include <gnuradio/flowgraph.hpp>
#include <assert.h>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace gr {

#define FLOWGRAPH_DEBUG 0

edge::~edge() {}

flowgraph::flowgraph() {}

flowgraph::~flowgraph() {}

template <class T>
static std::vector<T> unique_vector(std::vector<T> v)
{
    std::vector<T> result;
    std::insert_iterator<std::vector<T>> inserter(result, result.begin());

    sort(v.begin(), v.end());
    unique_copy(v.begin(), v.end(), inserter);
    return result;
}

void flowgraph::connect(const endpoint& src, const endpoint& dst)
{
    check_valid_port(src.block()->output_signature(), src.port());
    check_valid_port(dst.block()->input_signature(), dst.port());
    check_dst_not_used(dst);
    check_type_match(src, dst);

    // Alles klar, Herr Kommissar
    d_edges.push_back(edge(src, dst));
}

void flowgraph::disconnect(const endpoint& src, const endpoint& dst)
{
    for (edge_viter_t p = d_edges.begin(); p != d_edges.end(); p++) {
        if (src == p->src() && dst == p->dst()) {
            d_edges.erase(p);
            return;
        }
    }

    // std::stringstream msg;
    // msg << "cannot disconnect edge " << edge(src, dst) << ", not found";
    // throw std::invalid_argument(msg.str());
}

void flowgraph::validate()
{
    d_blocks = calc_used_blocks();

    //for (block_viter_t p = d_blocks.begin(); p != d_blocks.end(); p++) {
    for (auto &b : d_blocks)
    {
        std::vector<int> used_ports;
        int ninputs, noutputs;

        if (FLOWGRAPH_DEBUG)
            std::cout << "Validating block: " << b << std::endl;

        used_ports = calc_used_ports(b, true); // inputs
        ninputs = used_ports.size();
        check_contiguity(b, used_ports, true); // inputs

        used_ports = calc_used_ports(b, false); // outputs
        noutputs = used_ports.size();
        check_contiguity(b, used_ports, false); // outputs

        // if (!((*p)->check_topology(ninputs, noutputs))) {
        //     std::stringstream msg;
        //     msg << "check topology failed on " << (*p) << " using ninputs=" << ninputs
        //         << ", noutputs=" << noutputs;
        //     throw std::runtime_error(msg.str());
        // }

        // update the block alias
        std::map<std::string, int> name_count;
        // look in the map, see how many of that name exist
        // make the alias name + count;
        // increment the map
        int cnt;
        if (name_count.find(b->name()) == name_count.end()) {
            name_count[b->name()] = cnt = 0;
        } else {
            cnt = name_count[b->name()];
        }
        b->set_alias(b->name() + std::to_string(cnt));
        name_count[b->name()] = cnt+1;
        

    }
}

void flowgraph::clear()
{
    // Boost shared pointers will deallocate as needed
    d_blocks.clear();
    d_edges.clear();
}

void flowgraph::check_valid_port(gr::io_signature& sig, int port)
{
    std::stringstream msg;

    if (port < 0) {
        msg << "negative port number " << port << " is invalid";
        throw std::invalid_argument(msg.str());
    }

    if (port >= sig.n_streams()) {
        msg << "port number " << port << " exceeds max of " << sig.n_streams() - 1;
        throw std::invalid_argument(msg.str());
    }
}

void flowgraph::check_dst_not_used(const endpoint& dst)
{
    // A destination is in use if it is already on the edge list
    for (edge_viter_t p = d_edges.begin(); p != d_edges.end(); p++)
        if (p->dst() == dst) {
            std::stringstream msg;
            msg << "destination already in use by edge " << (*p);
            throw std::invalid_argument(msg.str());
        }
}

void flowgraph::check_type_match(const endpoint& src, const endpoint& dst)
{
    int src_size = src.block()->output_signature().sizeof_stream_item(src.port());
    int dst_size = dst.block()->input_signature().sizeof_stream_item(dst.port());

    if (src_size != dst_size) {
        std::stringstream msg;
        msg << "itemsize mismatch: " << src << " using " << src_size << ", " << dst
            << " using " << dst_size;
        throw std::invalid_argument(msg.str());
    }
}

block_vector_t flowgraph::calc_used_blocks()
{
    block_vector_t tmp;


    // Collect all blocks in the edge list
    for (edge_viter_t p = d_edges.begin(); p != d_edges.end(); p++) {
        tmp.push_back(p->src().block());
        tmp.push_back(p->dst().block());
    }

    return unique_vector<block_sptr>(tmp);
}

std::vector<int> flowgraph::calc_used_ports(block_sptr block, bool check_inputs)
{
    std::vector<int> tmp;

    // Collect all seen ports
    edge_vector_t edges = calc_connections(block, check_inputs);
    for (edge_viter_t p = edges.begin(); p != edges.end(); p++) {
        if (check_inputs == true)
            tmp.push_back(p->dst().port());
        else
            tmp.push_back(p->src().port());
    }

    return unique_vector<int>(tmp);
}

edge_vector_t flowgraph::calc_connections(block_sptr block, bool check_inputs)
{
    edge_vector_t result;

    for (edge_viter_t p = d_edges.begin(); p != d_edges.end(); p++) {
        if (check_inputs) {
            if (p->dst().block() == block)
                result.push_back(*p);
        } else {
            if (p->src().block() == block)
                result.push_back(*p);
        }
    }

    return result; // assumes no duplicates
}

void flowgraph::check_contiguity(block_sptr block,
                                 const std::vector<int>& used_ports,
                                 bool check_inputs)
{
    std::stringstream msg;

    gr::io_signature sig =
        check_inputs ? block->input_signature() : block->output_signature();

    int nports = used_ports.size();

    // TODO - make io_signature deal with optional ports -- port class??

    int min_ports = sig.n_streams();
    int max_ports = sig.n_streams();

    if (nports == 0 && min_ports == 0)
        return;

    if (nports < min_ports) {
        msg << block << ": insufficient connected "
            << (check_inputs ? "input ports " : "output ports ") << "(" << min_ports
            << " needed, " << nports << " connected)";
        throw std::runtime_error(msg.str());
    }

    if (nports > max_ports && max_ports != io_signature::IO_INFINITE) {
        msg << block << ": too many connected "
            << (check_inputs ? "input ports " : "output ports ") << "(" << max_ports
            << " allowed, " << nports << " connected)";
        throw std::runtime_error(msg.str());
    }

    if (used_ports[nports - 1] + 1 != nports) {
        for (int i = 0; i < nports; i++) {
            if (used_ports[i] != i) {
                msg << block << ": missing connection "
                    << (check_inputs ? "to input port " : "from output port ") << i;
                throw std::runtime_error(msg.str());
            }
        }
    }
}

block_vector_t flowgraph::calc_downstream_blocks(block_sptr block, int port)
{
    block_vector_t tmp;

    for (edge_viter_t p = d_edges.begin(); p != d_edges.end(); p++)
        if (p->src() == endpoint(block, port))
            tmp.push_back(p->dst().block());

    return unique_vector<block_sptr>(tmp);
}

block_vector_t flowgraph::calc_downstream_blocks(block_sptr block)
{
    block_vector_t tmp;

    for (edge_viter_t p = d_edges.begin(); p != d_edges.end(); p++)
        if (p->src().block() == block)
            tmp.push_back(p->dst().block());

    return unique_vector<block_sptr>(tmp);
}

edge_vector_t flowgraph::calc_upstream_edges(block_sptr block)
{
    edge_vector_t result;

    for (edge_viter_t p = d_edges.begin(); p != d_edges.end(); p++)
        if (p->dst().block() == block)
            result.push_back(*p);

    return result; // Assume no duplicates
}

bool flowgraph::has_block_p(block_sptr block)
{
    block_viter_t result;
    result = std::find(d_blocks.begin(), d_blocks.end(), block);
    return (result != d_blocks.end());
}

edge flowgraph::calc_upstream_edge(block_sptr block, int port)
{
    edge result;

    for (edge_viter_t p = d_edges.begin(); p != d_edges.end(); p++) {
        if (p->dst() == endpoint(block, port)) {
            result = (*p);
            break;
        }
    }

    return result;
}

std::vector<block_vector_t> flowgraph::partition()
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

block_vector_t flowgraph::calc_reachable_blocks(block_sptr block, block_vector_t& blocks)
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
void flowgraph::reachable_dfs_visit(block_sptr block, block_vector_t& blocks)
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
block_vector_t flowgraph::calc_adjacent_blocks(block_sptr block, block_vector_t& blocks)
{
    block_vector_t tmp;

    // Find any blocks that are inputs or outputs
    for (edge_viter_t p = d_edges.begin(); p != d_edges.end(); p++) {
        if (p->src().block() == block)
            tmp.push_back(p->dst().block());
        if (p->dst().block() == block)
            tmp.push_back(p->src().block());
    }

    return unique_vector<block_sptr>(tmp);
}

block_vector_t flowgraph::topological_sort(block_vector_t& blocks)
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

block_vector_t flowgraph::sort_sources_first(block_vector_t& blocks)
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

bool flowgraph::source_p(block_sptr block) { return calc_upstream_edges(block).empty(); }

void flowgraph::topological_dfs_visit(block_sptr block, block_vector_t& output)
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


} /* namespace gr */
