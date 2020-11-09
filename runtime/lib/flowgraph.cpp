#include <gnuradio/flowgraph.hpp>

namespace gr {

void flowgraph::set_scheduler(scheduler_sptr sched)
{
    d_schedulers = std::vector<scheduler_sptr>{ sched };

    // assign ids to the schedulers
    int idx = 1;
    for (auto s : d_schedulers) {
        s->set_id(idx++);
    }
}
void flowgraph::set_schedulers(std::vector<scheduler_sptr> sched)
{
    d_schedulers = sched;

    // assign ids to the schedulers
    int idx = 1;
    for (auto s : d_schedulers) {
        s->set_id(idx++);
    }
}
void flowgraph::add_scheduler(scheduler_sptr sched)
{
    d_schedulers.push_back(sched);
    // assign ids to the schedulers
    int idx = 1;
    for (auto s : d_schedulers) {
        s->set_id(idx++);
    }
}
void flowgraph::clear_schedulers() { d_schedulers.clear(); }
void flowgraph::partition(std::vector<domain_conf>& confs)
{
    d_fgmon = std::make_shared<flowgraph_monitor>(d_schedulers);
    // Create new subgraphs based on the partition configuration
    d_subgraphs.clear();

    // std::vector<std::tuple<graph,edge>> domain_crossings;
    std::vector<edge> domain_crossings;
    std::vector<domain_conf> crossing_confs;
    std::vector<scheduler_sptr> partition_scheds;
    std::vector<scheduler_sptr> crossing_scheds;

    std::map<nodeid_t, scheduler_sptr> block_to_scheduler_map;

    for (auto& conf : confs) {

        auto g = graph::make(); // create a new subgraph
        // Go through the blocks assigned to this scheduler
        // See whether they connect to the same graph or account for a domain crossing

        auto sched = conf.sched();   // std::get<0>(conf);
        auto blocks = conf.blocks(); // std::get<1>(conf);
        for (auto b : blocks)        // for each of the blocks in the tuple
        {
            // Store the block to scheduler mapping for later use
            block_to_scheduler_map[b->id()] = sched;

            for (auto input_port : b->input_stream_ports()) {
                auto edges = find_edge(input_port);
                // There should only be one edge connected to an input port
                // Crossings associated with the downstream port
                auto e = edges[0];
                auto other_block = e.src().node();

                // Is the other block in our current partition
                if (std::find(blocks.begin(), blocks.end(), other_block) !=
                    blocks.end()) {
                    g->connect(e.src(), e.dst(), e.buffer_factory(), e.buf_properties());
                } else {
                    // add this edge to the list of domain crossings
                    // domain_crossings.push_back(std::make_tuple(g,e));
                    domain_crossings.push_back(e);
                    crossing_confs.push_back(conf);

                    // Is this block connected to anything else in our current
                    // partition
                    bool connected = false; // TODO - handle orphan nodes
                }
            }
        }

        d_subgraphs.push_back(g);
        partition_scheds.push_back(sched);
    }

    int idx = 0;
    for (auto& conf : confs) {
        auto g = d_subgraphs[idx];

        // see that all the blocks in conf->blocks() are in g, and if not, add them as
        // orphan nodes

        for (auto b : conf.blocks()) // for each of the blocks in the tuple
        {
            bool connected = false;
            for (auto e : g->edges()) {
                if (e.src().node() == b || e.dst().node() == b) {
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

    // Now, let's set up domain adapters at the domain crossings
    // Several assumptions are being made now:
    //   1.  All schedulers running on the same processor
    //   2.  Outputs that cross domains can only be mapped one input
    //   3.  Fixed client/server relationship - limited configuration of DA

    std::map<scheduler_sptr, std::map<nodeid_t, neighbor_scheduler_info>>
        neighbor_map_per_scheduler;

    int crossing_index = 0;
    for (auto c : domain_crossings) {
        // Attach a domain adapter to the src and dst ports of the edge
        // auto g = std::get<0>(c);
        // auto e = std::get<1>(c);

        // Find the subgraph that holds src block
        graph_sptr src_block_graph = nullptr;
        for (auto g : d_subgraphs) {
            auto blocks = g->calc_used_nodes();
            if (std::find(blocks.begin(), blocks.end(), c.src().node()) != blocks.end()) {
                src_block_graph = g;
                break;
            }
        }

        // Find the subgraph that holds dst block
        graph_sptr dst_block_graph = nullptr;
        for (auto g : d_subgraphs) {
            auto blocks = g->calc_used_nodes();
            if (std::find(blocks.begin(), blocks.end(), c.dst().node()) != blocks.end()) {
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

        // Does the crossing have a specific domain adapter defined?
        domain_adapter_conf_sptr da_conf = nullptr;
        for (auto ec : conf.da_edge_confs()) {
            auto conf_edge = std::get<0>(ec);
            if (c == conf_edge) {
                da_conf = std::get<1>(ec);
                break;
            }
        }

        // else if defined: use the default defined for the domain
        if (!da_conf) {
            da_conf = conf.da_conf();
        }

        // else, use the default domain adapter configuration ??
        // TODO

        // use the conf to produce the domain adapters
        auto da_pair = da_conf->make_domain_adapter_pair(
            c.src().port(),
            c.dst().port(),
            "da_" + c.src().node()->alias() + "->" + c.dst().node()->alias());
        auto da_src = std::get<0>(da_pair);
        auto da_dst = std::get<1>(da_pair);


        // da_src->test();

        // Attach domain adapters to the src and dest blocks
        // domain adapters only have one port
        src_block_graph->connect(c.src(),
                                 node_endpoint(da_src, da_src->all_ports()[0]),
                                 c.buffer_factory(),
                                 c.buf_properties());
        dst_block_graph->connect(node_endpoint(da_dst, da_dst->all_ports()[0]),
                                 c.dst(),
                                 c.buffer_factory(),
                                 c.buf_properties());


        // Set the block id to "other scheduler" maps
        // This can/should be scheduler adapters, but use direct scheduler sptrs for now

        auto dst_block_id = c.dst().node()->id();
        auto src_block_id = c.src().node()->id();

        neighbor_map_per_scheduler[block_to_scheduler_map[dst_block_id]][dst_block_id]
            .set_upstream(block_to_scheduler_map[src_block_id], src_block_id);

        neighbor_map_per_scheduler[block_to_scheduler_map[src_block_id]][src_block_id]
            .add_downstream(block_to_scheduler_map[dst_block_id], dst_block_id);

        crossing_index++;
    }

    d_flat_subgraphs.clear();
    for (auto i = 0; i < partition_scheds.size(); i++) {
        d_flat_subgraphs.push_back(flat_graph::make_flat(d_subgraphs[i]));
        partition_scheds[i]->initialize(d_flat_subgraphs[i],
                                        d_fgmon,
                                        neighbor_map_per_scheduler[partition_scheds[i]]);
    }
}

void flowgraph::validate()
{
    gr_log_trace(_debug_logger, "validate()");
    d_fgmon = std::make_shared<flowgraph_monitor>(d_schedulers);

    d_flat_graph = flat_graph::make_flat(base());
    for (auto sched : d_schedulers)
        sched->initialize(d_flat_graph, d_fgmon);
}
void flowgraph::start()
{
    using namespace std::chrono_literals;
    gr_log_trace(_debug_logger, "start()");
    // Need thread synchronization for the schedulers - to know when they're done and
    // signal the other schedulers that might be connected

    d_fgmon->start();
    for (auto s : d_schedulers) {
        s->start();
    }
}
void flowgraph::stop()
{
    gr_log_trace(_debug_logger, "stop()");
    for (auto s : d_schedulers) {
        s->stop();
    }
    d_fgmon->stop();
}
void flowgraph::wait()
{
    gr_log_trace(_debug_logger, "wait()");
    for (auto s : d_schedulers) {
        s->wait();
    }
}

} // namespace gr
