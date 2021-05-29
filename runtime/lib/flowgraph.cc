#include <gnuradio/flowgraph.hh>
#include <gnuradio/graph_utils.hh>

#include <dlfcn.h>

namespace gr {


flowgraph::flowgraph()
{
    set_alias("flowgraph");

    // Dynamically load the module containing the default scheduler
    // Search path needs to be set correctly for qa in build dir
    void* handle = dlopen(("libnewsched-scheduler-" +
                              s_default_scheduler_name + ".so").c_str(),
                          RTLD_LAZY);
    if (!handle) {
        throw std::runtime_error("Unable to load default scheduler dynamically");
    }

    // TODO: Make the factory method more universal for any scheduler
    //  e.g. a json conf string or something generic interface
    std::shared_ptr<scheduler> (*factory)(const std::string&, size_t) =
        (std::shared_ptr<scheduler>(*)(const std::string&, size_t))dlsym(handle,
                                                                         "factory");

    // Instantiate the default scheduler
    d_default_scheduler = factory(s_default_scheduler_name, 32768);
    d_schedulers = { d_default_scheduler };
}

void flowgraph::set_scheduler(scheduler_sptr sched)
{
    if (d_default_scheduler_inuse) {
        d_default_scheduler_inuse = false;
    }
    d_schedulers = std::vector<scheduler_sptr>{ sched };

    // assign ids to the schedulers
    int idx = 1;
    for (auto s : d_schedulers) {
        s->set_id(idx++);
    }
}
void flowgraph::set_schedulers(std::vector<scheduler_sptr> sched)
{
    if (d_default_scheduler_inuse) {
        d_default_scheduler_inuse = false;
    }
    d_schedulers = sched;

    // assign ids to the schedulers
    int idx = 1;
    for (auto s : d_schedulers) {
        s->set_id(idx++);
    }
}
void flowgraph::add_scheduler(scheduler_sptr sched)
{
    if (d_default_scheduler_inuse) {
        d_default_scheduler_inuse = false;
        d_schedulers.clear();
    }
    d_schedulers.push_back(sched);
    // assign ids to the schedulers
    int idx = 1;
    for (auto s : d_schedulers) {
        s->set_id(idx++);
    }
}
void flowgraph::clear_schedulers() { d_schedulers.clear(); }

void flowgraph::check_connections(const graph_sptr& g)
{
    // Are all non-optional ports connected to something
    for (auto& node : g->calc_used_nodes()) {
        for (auto& port : node->output_ports()) {
            if (!port->optional() && port->connected_ports().size() == 0) {
                throw std::runtime_error("Nothing connected to " + port->alias());
            }
        }
        for (auto& port : node->input_ports()) {
            if (!port->optional()) {

                if (port->type() == port_type_t::STREAM) {

                    if (port->connected_ports().size() < 1) {
                        throw std::runtime_error("Nothing connected to " + port->alias());
                    } else if (port->connected_ports().size() > 1) {
                        throw std::runtime_error("More than 1 port connected to " +
                                                 port->alias());
                    }
                } else if (port->type() == port_type_t::MESSAGE) {
                    if (port->connected_ports().size() < 1) {
                        throw std::runtime_error("Nothing connected to " + port->alias());
                    }
                }
            }
        }
    }
}

void flowgraph::partition(std::vector<domain_conf>& confs)
{
    d_fgmon = std::make_shared<flowgraph_monitor>(d_schedulers);
    // Create new subgraphs based on the partition configuration

    check_connections(base());
    auto graph_part_info = graph_utils::partition(base(), d_schedulers, confs);

    d_flat_subgraphs.clear();
    for (auto& info : graph_part_info) {
        d_flat_subgraphs.push_back(flat_graph::make_flat(info.subgraph));
        info.scheduler->initialize(d_flat_subgraphs[d_flat_subgraphs.size() - 1],
                                   d_fgmon);
    }
}

void flowgraph::validate()
{
    GR_LOG_TRACE(_debug_logger, "validate()");
    d_fgmon = std::make_shared<flowgraph_monitor>(d_schedulers);

    d_flat_graph = flat_graph::make_flat(base());
    check_connections(d_flat_graph);

    for (auto sched : d_schedulers)
        sched->initialize(d_flat_graph, d_fgmon);

    _validated = true;
}

void flowgraph::start()
{
    if (!_validated) {
        validate();
    }

    GR_LOG_TRACE(_debug_logger, "start()");

    if (d_schedulers.empty()) {
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
