#include <gnuradio/runtime.hh>
#include <gnuradio/graph_utils.hh>
#include <gnuradio/flat_graph.hh>
#include <dlfcn.h>

namespace gr {
runtime::runtime()
{
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

void runtime::add_scheduler(std::pair<scheduler_sptr, std::vector<node_sptr>> conf)
{
    if (d_default_scheduler_inuse) {
        d_default_scheduler_inuse = false;
        d_schedulers.clear();
    }
    d_schedulers.push_back(std::get<0>(conf));
    d_blocks_per_scheduler.push_back(std::get<1>(conf));
    d_scheduler_confs.push_back(conf);
    // assign ids to the schedulers
    int idx = 1;
    for (auto s : d_schedulers) {
        s->set_id(idx++);
    }
}

void runtime::add_scheduler(scheduler_sptr sched)
{
    if (d_default_scheduler_inuse) {
        d_default_scheduler_inuse = false;
        d_schedulers.clear();
    }
    d_schedulers.push_back(sched);
    d_blocks_per_scheduler.push_back({});
    // assign ids to the schedulers
    int idx = 1;
    for (auto s : d_schedulers) {
        s->set_id(idx++);
    }
    for (auto s : d_runtime_proxies) {
        s->set_id(idx++);
    }
}

void runtime::add_proxy(runtime_proxy_sptr proxy)
{
    d_runtime_proxies.push_back(proxy);
    // assign ids to the schedulers
    int idx = 1;
    for (auto s : d_schedulers) {
        s->set_id(idx++);
    }
    for (auto s : d_runtime_proxies) {
        s->set_id(idx++);
    }
}

void runtime::initialize(flowgraph_sptr fg)
{
    flowgraph::check_connections(fg);
    auto _logger =
        logging::get_logger(fmt::format("runtime_init_{}", fg->name()), "default");
    GR_LOG_INFO(_logger, "initialize {}", d_schedulers.size());
    
    if (d_schedulers.size() == 1)
    {
        d_rtmon = std::make_shared<runtime_monitor>(
            d_schedulers, d_runtime_proxies, fg->alias());
        for (auto& p : d_runtime_proxies) {
            GR_LOG_DEBUG(_logger, ".");
            p->set_runtime_monitor(d_rtmon);
        }
        d_schedulers[0]->initialize(flat_graph::make_flat(fg),
                                   d_rtmon);
    }
    else
    {
        auto graph_part_info = graph_utils::partition(fg, d_scheduler_confs);
        d_rtmon = std::make_shared<runtime_monitor>(
            d_schedulers, d_runtime_proxies, fg->alias());
        for (auto& info : graph_part_info) {

            info.scheduler->initialize(flat_graph::make_flat(info.subgraph),
                                    d_rtmon);
        }
    }

    d_initialized = true;
}

void runtime::start() {
    if (!d_initialized) {
        throw new std::runtime_error("Runtime must be initialized prior to runtime start()");
    }
    d_rtmon->start();
}
void runtime::stop() {
    for (auto s : d_schedulers) {
        s->stop();
    }
    d_rtmon->stop();
}
void runtime::wait() {
    for (auto s : d_schedulers) {
        s->wait();
    }
}

void runtime::run() {
    start();
    wait();
}

} // namespace gr
