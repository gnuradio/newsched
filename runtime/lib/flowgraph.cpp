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
    for (auto& info : graph_part_info) {
        d_flat_subgraphs.push_back(flat_graph::make_flat(info.subgraph));
        info.scheduler->initialize(
            d_flat_subgraphs[d_flat_subgraphs.size() - 1], d_fgmon);
    }
}

void flowgraph::validate()
{
    GR_LOG_TRACE(_debug_logger, "validate()");
    d_fgmon = std::make_shared<flowgraph_monitor>(d_schedulers);

    d_flat_graph = flat_graph::make_flat(base());
    for (auto sched : d_schedulers)
        sched->initialize(d_flat_graph, d_fgmon);
}
void flowgraph::start()
{
    GR_LOG_TRACE(_debug_logger, "start()");

    if (d_schedulers.empty())
    {
        GR_LOG_ERROR(_logger, "No Scheduler Specified.");
    }

    d_fgmon->start();
    for (auto s : d_schedulers) {
        s->start();
    }
}
void flowgraph::stop()
{
    GR_LOG_TRACE(_debug_logger, "stop()");
    for (auto s : d_schedulers) {
        s->stop();
    }
    d_fgmon->stop();
}
void flowgraph::wait()
{
    GR_LOG_TRACE(_debug_logger, "wait()");
    for (auto s : d_schedulers) {
        s->wait();
    }
}
void flowgraph::run()
{
    start();
    wait();
}

} // namespace gr
