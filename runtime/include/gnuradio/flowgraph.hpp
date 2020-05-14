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
#include "block.hpp"
#include "io_signature.hpp"
#include <iostream>

namespace gr {

/*!
 * \brief Class representing a specific input or output graph endpoint
 * \ingroup internal
 */
class GR_RUNTIME_API endpoint
{
private:
    block_sptr d_block;
    int d_port;

public:
    endpoint() : d_block(), d_port(0) {}
    endpoint(block_sptr block, int port)
    {
        d_block = block;
        d_port = port;
    }
    block_sptr block() const { return d_block; }
    int port() const { return d_port; }
    std::string identifier() const
    {
        return d_block->alias() + ":" + std::to_string(d_port);
    };

    bool operator==(const endpoint& other) const;
};

inline bool endpoint::operator==(const endpoint& other) const
{
    return (d_block == other.d_block && d_port == other.d_port);
}

inline std::ostream& operator<<(std::ostream& os, const endpoint endp)
{
    os << endp.identifier();
    return os;
}


// Hold vectors of gr::endpoint objects
typedef std::vector<endpoint> endpoint_vector_t;
typedef std::vector<endpoint>::iterator endpoint_viter_t;

/*!
 *\brief Class representing a connection between to graph endpoints
 */
class GR_RUNTIME_API edge
{
public:
    edge() : d_src(), d_dst(){};
    edge(const endpoint& src, const endpoint& dst) : d_src(src), d_dst(dst) {}
    ~edge();

    const endpoint& src() const { return d_src; }
    const endpoint& dst() const { return d_dst; }
    std::string identifier() const
    {
        return d_src.identifier() + "->" + d_dst.identifier();
    }

    unsigned int itemsize() const
    {
        return d_src.block()->output_signature().sizeof_stream_item(d_src.port());
    }

private:
    endpoint d_src;
    endpoint d_dst;
};

inline std::ostream& operator<<(std::ostream& os, const edge edge)
{
    os << edge.identifier();
    return os;
}

// Hold vectors of gr::edge objects
typedef std::vector<edge> edge_vector_t;
typedef std::vector<edge>::iterator edge_viter_t;


/*!
 * \brief Class representing a directed, acyclic graph of basic blocks
 * \ingroup internal
 */
class GR_RUNTIME_API flowgraph
{
public:
    flowgraph();
    typedef std::shared_ptr<flowgraph> sptr;
    virtual ~flowgraph();

    /*!
     * \brief Connect two endpoints
     * \details
     * Checks the validity of both endpoints, and whether the
     * destination is unused so far, then adds the edge to the internal list of
     * edges.
     */
    void connect(const endpoint& src, const endpoint& dst);

    /*!
     * \brief Disconnect two endpoints
     */
    void disconnect(const endpoint& src, const endpoint& dst);


    /*!
     * \brief Validate flow graph
     * \details
     * Gathers all used blocks, checks the contiguity of all connected in- and
     * outputs, and calls the check_topology method of each block.
     */
    void validate();

    /*!
     * \brief Clear existing flowgraph
     */
    void clear();

    /*!
     * \brief Get vector of edges
     */
    const edge_vector_t& edges() const { return d_edges; }

    /*!
     * \brief calculates all used blocks in a flow graph
     * \details
     * Iterates over all message edges and stream edges, noting both endpoints in a
     * vector.
     *
     * \return a unique vector of used blocks
     */
    block_vector_t calc_used_blocks();

    /*!
     * \brief topologically sort blocks
     * \details
     * Uses depth-first search to return a sorted vector of blocks
     *
     * \return toplogically sorted vector of blocks.  All the sources come first.
     */
    block_vector_t topological_sort(block_vector_t& blocks);

    /*!
     * \brief Calculate vector of disjoint graph partions
     * \return vector of disjoint vectors of topologically sorted blocks
     */
    std::vector<block_vector_t> partition();

    // bool is_flat() { return true; };
    // sptr flatten() { };  // make a flowgraph object that is flattened

    edge find_edge(block_sptr b, int port_index, block::io io_type)
    {
        for (auto& e : edges())
        {
            endpoint tmp(b,port_index);
            if (io_type == block::io::INPUT)
            {
                if (e.dst().identifier() == tmp.identifier())
                {
                    return e;
                }
            }
            else
            {
                if (e.src().identifier() == tmp.identifier())
                {
                    return e;
                }
            }
        }

        throw std::invalid_argument( "edge not found" );
    }

protected:
    block_vector_t d_blocks;
    edge_vector_t d_edges;

    std::vector<int> calc_used_ports(block_sptr block, bool check_inputs);
    block_vector_t calc_downstream_blocks(block_sptr block, int port);
    edge_vector_t calc_upstream_edges(block_sptr block);
    bool has_block_p(block_sptr block);
    edge calc_upstream_edge(block_sptr block, int port);

private:
    void check_valid_port(gr::io_signature& sig, int port);
    void check_dst_not_used(const endpoint& dst);
    void check_type_match(const endpoint& src, const endpoint& dst);
    edge_vector_t calc_connections(block_sptr block,
                                   bool check_inputs); // false=use outputs
    void check_contiguity(block_sptr block,
                          const std::vector<int>& used_ports,
                          bool check_inputs);

    block_vector_t calc_downstream_blocks(block_sptr block);
    block_vector_t calc_reachable_blocks(block_sptr blk,
                                               block_vector_t& blocks);
    void reachable_dfs_visit(block_sptr blk, block_vector_t& blocks);
    block_vector_t calc_adjacent_blocks(block_sptr blk,
                                              block_vector_t& blocks);
    block_vector_t sort_sources_first(block_vector_t& blocks);
    bool source_p(block_sptr block);
    void topological_dfs_visit(block_sptr blk, block_vector_t& output);
};

typedef flowgraph::sptr flowgraph_sptr;
}

#endif