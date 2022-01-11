#include <gnuradio/flowgraph.hh>
#include <gnuradio/graph_utils.hh>

#include <dlfcn.h>
#include <httplib.h>

namespace gr {


flowgraph::flowgraph(const std::string& name, bool secondary) : s_secondary(secondary)
{
    set_alias(name);

    // Dynamically load the module containing the default scheduler
    // Search path needs to be set correctly for qa in build dir
    void* handle = dlopen(
        ("libnewsched-scheduler-" + s_default_scheduler_name + ".so").c_str(), RTLD_LAZY);
    if (!handle) {
        throw std::runtime_error("Unable to load default scheduler dynamically");
    }

    std::shared_ptr<scheduler> (*factory)(const std::string&) =
        (std::shared_ptr<scheduler>(*)(const std::string&))dlsym(handle, "factory");
    // Instantiate the default scheduler
    d_default_scheduler = factory("{name: nbt, buffer_size: 32768}");
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
    for (auto s : d_fgm_proxies) {
        s->set_id(idx++);
    }
}
void flowgraph::add_fgm_proxy(fgm_proxy_sptr fgm_proxy)
{
    d_fgm_proxies.push_back(fgm_proxy);
    // assign ids to the schedulers
    int idx = 1;
    for (auto s : d_schedulers) {
        s->set_id(idx++);
    }
    for (auto s : d_fgm_proxies) {
        s->set_id(idx++);
    }
}
void flowgraph::clear_schedulers() { d_schedulers.clear(); }

size_t get_port_itemsize(port_sptr port)
{
    size_t size = 0;
    if (port->connected_ports().size() > 0) {
        auto cp = port->connected_ports()[0];
        // use data_size since this includes vector sizing
        size = cp->data_size();
    }
    return size;
}

std::string get_port_format_descriptor(port_sptr port)
{
    std::string fd = "";
    if (port->connected_ports().size() > 0) {
        auto cp = port->connected_ports()[0];
        // use data_size since this includes vector sizing
        fd = cp->format_descriptor();
    }
    return fd;
}


void flowgraph::check_connections(const graph_sptr& g)
{
    // Are all non-optional ports connected to something
    for (auto& node : g->calc_used_nodes()) {
        if (node) {
            for (auto& port : node->output_ports()) {
                // if (!port->optional() && port->connected_ports().size() == 0) {
                //     throw std::runtime_error("Nothing connected [1] to " + node->name()
                //     + ": " + port->name());
                // }
            }
            for (auto& port : node->input_ports()) {
                if (!port->optional()) {

                    if (port->type() == port_type_t::STREAM) {

                        if (port->connected_ports().size() < 1) {
                            // throw std::runtime_error("Nothing connected [2] to " +
                            // node->name() + ": " + port->name());
                        } else if (port->connected_ports().size() > 1) {
                            throw std::runtime_error("More than 1 port connected to " +
                                                     port->alias());
                        }
                    } else if (port->type() == port_type_t::MESSAGE) {
                        if (port->connected_ports().size() < 1) {
                            throw std::runtime_error("Nothing connected [3] to " +
                                                     node->name() + ": " + port->name());
                        }
                    }
                }
            }
        }
    }

    // Iteratively check the flowgraph for 0 size ports and adjust
    bool updated_sizes = true;
    while (updated_sizes) {
        updated_sizes = false;
        for (auto& node : g->calc_used_nodes()) {
            for (auto& port : node->output_ports()) {
                if (port->itemsize() == 0) {
                    // we need to propagate the itemsize from whatever it is connected to
                    auto newsize = get_port_itemsize(port);
                    auto newfd = get_port_format_descriptor(port);
                    port->set_itemsize(newsize);
                    port->set_format_descriptor(newfd);
                    updated_sizes = newsize > 0;
                }
            }
            for (auto& port : node->input_ports()) {
                if (port->itemsize() == 0) {
                    // we need to propagate the itemsize from whatever it is connected to
                    auto newsize = get_port_itemsize(port);
                    auto newfd = get_port_format_descriptor(port);
                    port->set_itemsize(newsize);
                    port->set_format_descriptor(newfd);
                    updated_sizes = newsize > 0;
                }
            }
        }
    }

    // Set any remaining 0 size ports to something
    size_t newsize = sizeof(gr_complex); // why not, does it matter
    for (auto& node : g->calc_used_nodes()) {
        for (auto& port : node->output_ports()) {
            if (port->itemsize() == 0) {
                port->set_itemsize(newsize);
                port->set_format_descriptor("Zf");
            }
        }
        for (auto& port : node->input_ports()) {
            if (port->itemsize() == 0) {
                port->set_itemsize(newsize);
                port->set_format_descriptor("Zf");
            }
        }
    }
}

void flowgraph::partition(std::vector<domain_conf>& confs)
{
    // the schedulers contained in confs should be complete with the flowgraph
    // So we can add them here
    clear_schedulers();
    for (auto& conf : confs) {
        if (!conf.execution_host()) { // <-- this scheduler is running locally
            add_scheduler(conf.sched());
        }
    }

    d_fgmon = std::make_shared<flowgraph_monitor>(d_schedulers, d_fgm_proxies);
    // Create new subgraphs based on the partition configuration

    check_connections(base());
    auto graph_part_info = graph_utils::partition(base(), confs);

    int conf_index = 0;
    for (auto& info : graph_part_info) {
        auto flattened_graph = flat_graph::make_flat(info.subgraph);

        if (auto host = confs[conf_index].execution_host()) {
            // Serialize and reprogram the flattened graph on the remote side
            httplib::Client cli("http://" + host->ipaddr() + ":" + std::to_string(host->port()) );
            // 1. Create flowgraph
            auto res = cli.Get("/flowgraph/foo/create");

            // 2. Create Blocks
            for (auto& b : confs[conf_index].blocks())
            {
                // curl -v -H "Content-Type: application/json" POST      -d '{"module": "blocks", "id": "vector_source_c", "parameters": {"data": [1,2,3,4,5], "repeat": false }}' http://127.0.0.1:8000/block/src/create
                cli.Post(("/block/" + b->alias() + "/create").c_str(), std::static_pointer_cast<block>(b)->to_json().c_str(), "application/json");
            }

            // 3. Connect Blocks (or add edges)

            // 4. Create Scheduler


        } else {
            info.scheduler->initialize(flattened_graph, d_fgmon);
        }

        conf_index++;
    }
    _validated = true;
}

void flowgraph::validate()
{
    GR_LOG_TRACE(_debug_logger, "validate()");
    d_fgmon = std::make_shared<flowgraph_monitor>(d_schedulers, d_fgm_proxies);
    for (auto& p : d_fgm_proxies) {
        p->set_fgm(d_fgmon);
    }

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

    d_fgmon->start(alias());
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
