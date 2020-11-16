#include <gnuradio/flowgraph.hpp>
#include <gnuradio/graph_utils.hpp>

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

    auto graph_part_info = graph_utils::partition(base(), d_schedulers, confs);

    d_flat_subgraphs.clear();
    for (size_t i = 0; i < graph_part_info.subgraphs.size(); i++) {
        d_flat_subgraphs.push_back(flat_graph::make_flat(graph_part_info.subgraphs[i]));
        graph_part_info.partition_scheds[i]->initialize(
            d_flat_subgraphs[i],
            d_fgmon,
            graph_part_info
                .neighbor_map_per_scheduler[graph_part_info.partition_scheds[i]]);
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
