#pragma once

#include <gnuradio/block.hpp>
#include <gnuradio/graph.hpp>

namespace gr {

/**
 * @brief Endpoint consisting only of Blocks
 *
 */
class block_endpoint : public node_endpoint
{
private:
    block_sptr d_block;

public:
    block_endpoint(block_sptr block, port_sptr port)
        : node_endpoint(block, port), d_block(block)
    {
    }

    block_endpoint(const node_endpoint& n) : node_endpoint(n) {}

    block_sptr block() { return std::dynamic_pointer_cast<gr::block>(this->node()); }
};

/**
 * @brief Flattened Graph class
 *
 * Graph that contains only blocks, and all the topological logic to determine cycles and
 * connectivity
 *
 */
class flat_graph : public graph
{
    static constexpr const char* BLOCK_COLOR_KEY = "color";
    enum vcolor { WHITE, GREY, BLACK };
    enum io { INPUT, OUTPUT };

public:
    void clear();
    flat_graph() {}
    // typedef std::shared_ptr<flat_graph> sptr;
    virtual ~flat_graph();

    block_vector_t calc_used_blocks()
    {
        block_vector_t tmp;

        // Collect all blocks in the edge list
        for (auto& p : edges()) {
            // if both ends of the edge belong to this graph
            if (std::find(_nodes.begin(), _nodes.end(), p->src().node()) !=
                    _nodes.end() &&
                std::find(_nodes.begin(), _nodes.end(), p->dst().node()) !=
                    _nodes.end()) {

                auto src_ptr = std::dynamic_pointer_cast<block>(p->src().node());
                auto dst_ptr = std::dynamic_pointer_cast<block>(p->dst().node());

                if (src_ptr != nullptr) {
                    tmp.push_back(src_ptr);
                }
                if (dst_ptr != nullptr) {
                    tmp.push_back(dst_ptr);
                }
            }
        }

        return unique_vector<block_sptr>(tmp);
    }

    static std::shared_ptr<flat_graph> make_flat(graph_sptr g)
    {
        // FIXME: Actually do the flattening
        // for now assume it is already flat, and just cast things
        auto fg = std::make_shared<flat_graph>();
        for (auto e : g->edges()) {
            // connect only if both sides of the edge are in this graph
            if (std::find(g->nodes().begin(), g->nodes().end(), e->src().node()) !=
                    g->nodes().end() &&
                std::find(g->nodes().begin(), g->nodes().end(), e->dst().node()) !=
                    g->nodes().end()) {
                fg->connect(e->src(), e->dst())
                    ->set_custom_buffer(e->buf_properties());
            } else { // edge is a pathway into another domain
                fg->add_edge(e);
            }
        }
        for (auto o : g->orphan_nodes()) {
            fg->add_orphan_node(o);
        }

        return fg;
    }

    block_vector_t calc_downstream_blocks(block_sptr block, port_sptr port);

protected:
    block_vector_t d_blocks;

    port_vector_t calc_used_ports(block_sptr block, bool check_inputs);
    edge_vector_t calc_upstream_edges(block_sptr block);
    bool has_block_p(block_sptr block);

private:
    edge_vector_t calc_connections(block_sptr block,
                                   bool check_inputs); // false=use outputs
    block_vector_t calc_downstream_blocks(block_sptr block);
    block_vector_t calc_reachable_blocks(block_sptr blk, block_vector_t& blocks);
    void reachable_dfs_visit(block_sptr blk, block_vector_t& blocks);
    block_vector_t calc_adjacent_blocks(block_sptr blk, block_vector_t& blocks);
    block_vector_t sort_sources_first(block_vector_t& blocks);
    bool source_p(block_sptr block);
    void topological_dfs_visit(block_sptr blk, block_vector_t& output);

    std::vector<block_vector_t> partition();
    block_vector_t topological_sort(block_vector_t& blocks);
};

typedef std::shared_ptr<flat_graph> flat_graph_sptr;

} // namespace gr
