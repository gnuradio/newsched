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
}

void runtime::initialize(flowgraph_sptr fg)
{
    d_fgmon = std::make_shared<flowgraph_monitor>(d_schedulers);
    flowgraph::check_connections(fg);

    if (d_schedulers.size() == 1)
    {
        d_schedulers[0]->initialize(flat_graph::make_flat(fg),
                                   d_fgmon);
    }
    else
    {
        auto graph_part_info = graph_utils::partition(fg, d_scheduler_confs);
        for (auto& info : graph_part_info) {
            info.scheduler->initialize(flat_graph::make_flat(info.subgraph),
                                    d_fgmon);
        }
    }

    d_initialized = true;
}

void runtime::start() {
    if (!d_initialized) {
        throw new std::runtime_error("Runtime must be initialized prior to runtime start()");
    }
    d_fgmon->start();
    for (auto s : d_schedulers) {
        s->start();
    }
}
void runtime::stop() {
    for (auto s : d_schedulers) {
        s->stop();
    }
    d_fgmon->stop();
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
