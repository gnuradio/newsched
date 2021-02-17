#include "buffer_management.hpp"
#include "scheduler_st.hpp"

namespace gr {

namespace schedulers {


scheduler_st::scheduler_st(const std::string name, const unsigned int fixed_buf_size)
    : scheduler(name), s_fixed_buf_size(fixed_buf_size)
{
    _default_buf_factory = simplebuffer::make;
}

void scheduler_st::initialize(flat_graph_sptr fg,
                              flowgraph_monitor_sptr fgmon,
                              neighbor_interface_map block_sched_map)
{
    for (auto& b : fg->calc_used_blocks()) {
        b->set_scheduler(base());
    }

    // if (fg->is_flat())  // flatten
    auto bufman = std::make_shared<buffer_manager>(s_fixed_buf_size);
    bufman->initialize_buffers(fg, _default_buf_factory, _default_buf_properties);

    for (auto& b : fg->calc_used_nodes()) {

        for (auto& p : b->all_ports()) {
            p->set_parent_intf(base()); // give a shared pointer to the scheduler class
        }
    }

    _thread = thread_wrapper::make(
        name(), id(), fg->calc_used_blocks(), block_sched_map, bufman, fgmon);
}

void scheduler_st::start() { _thread->start(); }
void scheduler_st::stop() { _thread->stop(); }
void scheduler_st::wait() { _thread->wait(); }
void scheduler_st::run() { _thread->run(); }


} // namespace schedulers
} // namespace gr
