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
