#include <gnuradio/graph_utils.hh>

#include <gnuradio/block.hh>
#include <gnuradio/buffer_net_zmq.hh>
#include <gnuradio/domain.hh>
namespace gr {


graph_partition_info_vec graph_utils::partition(graph_sptr input_graph,
                                                std::vector<domain_conf>& confs)
{
    graph_partition_info_vec ret;
    std::map<scheduler_sptr, size_t> sched_index_map;

    edge_vector_t domain_crossings;
    std::vector<domain_conf> crossing_confs;

    std::map<nodeid_t, scheduler_sptr> block_to_scheduler_map;

    for (auto& conf : confs) {

        auto g = graph::make(); // create a new subgraph
        // Go through the blocks assigned to this scheduler
        // See whether they connect to the same graph or account for a domain crossing

        graph_partition_info part_info;
        auto sched = conf.sched();   // std::get<0>(conf);
        auto blocks = conf.blocks(); // std::get<1>(conf);
        for (auto b : blocks)        // for each of the blocks in the tuple
        {
            // Store the block to scheduler mapping for later use
            block_to_scheduler_map[b->id()] = sched;

            for (auto input_port : b->input_ports()) {
                auto edges = input_graph->find_edge(input_port);
                // There should only be one edge connected to an input port
                // Crossings associated with the downstream port
                if (edges.size()) {
                    auto e = edges[0];
                    auto other_block = e->src().node();

                    // Is the other block in our current partition
                    if (std::find(blocks.begin(), blocks.end(), other_block) !=
                        blocks.end()) {
                        g->connect(e->src(), e->dst())
                            ->set_custom_buffer(e->buf_properties());
                    } else {
                        // add this edge to the list of domain crossings
                        // domain_crossings.push_back(std::make_tuple(g,e));
                        domain_crossings.push_back(e);
                        crossing_confs.push_back(conf);
                    }
                }
            }
        }

        sched_index_map[sched] = ret.size();

        part_info.subgraph = g;
        part_info.scheduler = sched;
        // neighbor_map is populated below
        ret.push_back(part_info);
    }

    int idx = 0;
    for (auto& conf : confs) {
        auto g = ret[idx].subgraph;

        // see that all the blocks in conf->blocks() are in g, and if not, add them as
        // orphan nodes

        for (auto b : conf.blocks()) // for each of the blocks in the tuple
        {
            bool connected = false;
            for (auto e : g->edges()) {
                if (e->src().node() == b || e->dst().node() == b) {
                    connected = true;
                    break;
                }
            }

            if (!connected) {
                g->add_orphan_node(b);
            }
        }

        idx++;
    }

    // For the crossing, make sure the edge that crosses the domain is included

    int crossing_index = 0;
    for (auto c : domain_crossings) {

        // Find the subgraph that holds src block
        graph_sptr src_block_graph = nullptr;
        for (auto info : ret) {
            auto g = info.subgraph;
            auto blocks = g->calc_used_nodes();
            if (std::find(blocks.begin(), blocks.end(), c->src().node()) !=
                blocks.end()) {
                src_block_graph = g;
                break;
            }
        }

        // Find the subgraph that holds dst block
        graph_sptr dst_block_graph = nullptr;
        for (auto info : ret) {
            auto g = info.subgraph;
            auto blocks = g->calc_used_nodes();
            if (std::find(blocks.begin(), blocks.end(), c->dst().node()) !=
                blocks.end()) {
                dst_block_graph = g;
                break;
            }
        }

        if (!src_block_graph || !dst_block_graph) {
            throw std::runtime_error("Cannot find both sides of domain crossing");
        }

        // Crossings are associated with the downstream port
        // so the execution host here corresponds with the dst block
        if (auto host = crossing_confs[crossing_index].execution_host()) {
            // In this case we need to duplicate the edge and add a custom buffer
            // to each side
            // Placeholder edge - perhaps needs some additional designation
            auto upstream_edge = edge::make(c->src(), c->dst());
            auto downstream_edge = edge::make(c->src(), c->dst());
            // upstream_edge->set_custom_buffer(
            //     buffer_net_zmq_properties::make(host->ipaddr(), host->port()));
            // downstream_edge->set_custom_buffer(
            //     buffer_net_zmq_properties::make(host->ipaddr(), host->port()));
            src_block_graph->add_edge(upstream_edge);
            dst_block_graph->add_edge(downstream_edge);
        } else {
            src_block_graph->add_edge(c);
            dst_block_graph->add_edge(c);
        }

        crossing_index++;
    }

    return ret;
}
} // namespace gr
