#include <gnuradio/domain.hpp>
#include <gnuradio/domain_adapter_direct.hpp>
#include <gnuradio/graph_utils.hpp>
#include <gnuradio/scheduler.hpp>
#include <gnuradio/schedulers/st/scheduler_st.hpp>

namespace gr {
namespace schedulers {
class scheduler_mt : public scheduler
{
private:
    // std::vector<thread_wrapper::ptr> _threads;
    std::vector<scheduler_sptr> _st_scheds;
    const int s_fixed_buf_size;
    std::map<nodeid_t, scheduler_sptr> _block_thread_map;

public:
    typedef std::shared_ptr<scheduler_mt> sptr;
    static sptr make(const std::string name = "multi_threaded",
                     const unsigned int fixed_buf_size = 8192)
    {
        return std::make_shared<scheduler_mt>(name, fixed_buf_size);
    }
    scheduler_mt(const std::string name = "multi_threaded",
                 const unsigned int fixed_buf_size = 8192)
        : scheduler(name), s_fixed_buf_size(fixed_buf_size)
    {
        _default_buf_factory = simplebuffer::make;
    }
    ~scheduler_mt(){};

    void push_message(scheduler_message_sptr msg)
    { 
        _block_thread_map[msg->blkid()]->push_message(msg);
    }
    bool pop_message(scheduler_message_sptr& msg)
    {
        return _block_thread_map[msg->blkid()]->pop_message(msg);
    }

    void initialize(flat_graph_sptr fg,
                    flowgraph_monitor_sptr fgmon,
                    neighbor_interface_map block_sched_map)
    {
        auto da_conf = domain_adapter_direct_conf::make(buffer_preference_t::UPSTREAM);


        std::vector<scheduler_sptr> scheds;
        domain_conf_vec dconf;
        //  Partition the flowgraph according to how blocks are specified in groups
        //  For now, one Thread Per Block

        // If a block already has a domain adapter attached to it,
        // leave it in the domain conf
        auto blocks = fg->calc_used_blocks();
        int idx = 0;
        for (auto& b : blocks) { // domain adapters don't show up as blocks
            auto st_sched =
                scheduler_st::make(name() + std::to_string(idx), s_fixed_buf_size);
            scheds.push_back(st_sched);
            std::vector<node_sptr> node_vec;
            node_vec.push_back(b);

            // If b has connections to any domain adapters, include them in this scheduler
            for (auto& p : b->input_ports()) {
                for (auto& ed : fg->find_edge(p)) {
                    auto da_cast =
                        std::dynamic_pointer_cast<domain_adapter>(ed.src().node());
                    if (da_cast != nullptr) {
                        node_vec.push_back(ed.src().node());
                    }
                }
            }
            for (auto& p : b->output_ports()) {
                for (auto& ed : fg->find_edge(p)) {
                    auto da_cast =
                        std::dynamic_pointer_cast<domain_adapter>(ed.dst().node());
                    if (da_cast != nullptr) {
                        node_vec.push_back(ed.dst().node());
                    }
                }
            }

            dconf.push_back(domain_conf(st_sched, node_vec, da_conf));
            idx++;
        }

        fgmon->replace_scheduler(base(), scheds);

        auto partition_info = graph_utils::partition(fg, scheds, dconf, block_sched_map);

        for (auto& info : partition_info) {
            auto flattened = flat_graph::make_flat(info.subgraph);
            info.scheduler->initialize(
                flattened, fgmon, info.neighbor_map);
            _st_scheds.push_back(info.scheduler);

            for (auto& b: flattened->calc_used_nodes())
            {
                _block_thread_map[b->id()] = info.scheduler;
            }
            
        }


        // if (fg->is_flat())  // flatten
    }
    void start()
    {
        for (const auto& thd : _st_scheds) {
            thd->start();
        }
    }
    void stop()
    {
        for (const auto& thd : _st_scheds) {
            thd->stop();
        }
    }
    void wait()
    {
        for (const auto& thd : _st_scheds) {
            thd->wait();
        }
    }
    void run()
    {
        for (const auto& thd : _st_scheds) {
            thd->start();
        }
        for (const auto& thd : _st_scheds) {
            thd->wait();
        }
    }
};
} // namespace schedulers
} // namespace gr