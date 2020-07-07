#pragma once

#include <gnuradio/graph.hpp>

// All the things that happen to a graph once it is flat
// Works only with block_sptr and block_edges
namespace gr {

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
        for (edge_viter_t p = _edges.begin(); p != _edges.end(); p++) {
            tmp.push_back(static_cast<block_endpoint>(p->src()).block());
            tmp.push_back(static_cast<block_endpoint>(p->dst()).block());
        }

        return unique_vector<block_sptr>(tmp);
    }

    static std::shared_ptr<flat_graph> make_flat(graph_sptr g)
    {
        // for now assume it is already flat, and just cast things
        std::shared_ptr<flat_graph> fg = std::shared_ptr<flat_graph>(new flat_graph());
        for (auto e : g->edges()) {
            fg->connect(e.src(), e.dst());
        }

        return fg;
    }

    std::vector<edge> find_edge(port_sptr port)
    {
        std::vector<edge> ret;
        for (auto& e : edges()) {
            if (e.src().port() == port)
                ret.push_back(e);

            if (e.dst().port() == port)
                ret.push_back(e);
        }

        if (ret.empty())
            throw std::invalid_argument("edge not found");

        return ret;
    }

protected:
    block_vector_t d_blocks;
    // edge_vector_t d_edges;


    port_vector_t calc_used_ports(block_sptr block, bool check_inputs);
    block_vector_t calc_downstream_blocks(block_sptr block, port_sptr port);
    edge_vector_t calc_upstream_edges(block_sptr block);
    bool has_block_p(block_sptr block);

private:
    void check_valid_port(block_sptr block, port_sptr port);
    void check_dst_not_used(const block_endpoint& dst);
    void check_type_match(const block_endpoint& src, const block_endpoint& dst);
    edge_vector_t calc_connections(block_sptr block,
                                   bool check_inputs); // false=use outputs
    void
    check_contiguity(block_sptr block, const port_vector_t used_ports, bool check_inputs);

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