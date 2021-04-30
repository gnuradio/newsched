#include <gnuradio/schedulers/mt/scheduler_mt.hpp>

namespace gr {
namespace schedulers {

void scheduler_mt::push_message(scheduler_message_sptr msg)
{
    // Use 0 for blkid all threads
    if (msg->blkid() == 0) {
        for (auto element : _block_thread_map) {
            auto thd = element.second;
            thd->push_message(msg);
        }
    } else {
        _block_thread_map[msg->blkid()]->push_message(msg);
    }
}

void scheduler_mt::add_block_group(const std::vector<block_sptr>& blocks, const std::string& name, const std::vector<unsigned int>& affinity_mask)
{
    _block_groups.push_back(std::move(block_group_properties(blocks, name, affinity_mask)));
}

void scheduler_mt::initialize(flat_graph_sptr fg,
                              flowgraph_monitor_sptr fgmon)
{
    for (auto& b : fg->calc_used_blocks()) {
        b->set_scheduler(base());
    }

    auto bufman = std::make_shared<buffer_manager>(s_fixed_buf_size);
    bufman->initialize_buffers(fg, _default_buf_properties);


    //  Partition the flowgraph according to how blocks are specified in groups
    //  By default, one Thread Per Block

    auto blocks = fg->calc_used_blocks();

    // look at our block groups, create confs and remove from blocks
    for (auto& bg : _block_groups) {
        std::vector<block_sptr> blocks_for_this_thread;

        if (bg.blocks().size()) {
            auto t = thread_wrapper::make(
                 id(), bg, bufman, fgmon);
            _threads.push_back(t);

            std::vector<node_sptr> node_vec;
            for (auto& b : bg.blocks()) { // domain adapters don't show up as blocks
                auto it = std::find(blocks.begin(), blocks.end(), b);
                if (it != blocks.end()) {
                    blocks.erase(it);
                }

                node_vec.push_back(b);

                for (auto& p : b->all_ports()) {
                    p->set_parent_intf(t); // give a shared pointer to the scheduler class
                }
                _block_thread_map[b->id()] = t;
            }
        }
    }

    // For the remaining blocks that weren't in block groups
    for (auto& b : blocks) {

        std::vector<node_sptr> node_vec;
        node_vec.push_back(b);

        auto t =
            thread_wrapper::make(id(), block_group_properties({b}), bufman, fgmon);
        _threads.push_back(t);

        for (auto& p : b->all_ports()) {
            p->set_parent_intf(t); // give a shared pointer to the scheduler class
        }

        _block_thread_map[b->id()] = t;
    }
}

void scheduler_mt::start()
{
    for (const auto& thd : _threads) {
        thd->start();
    }
}
void scheduler_mt::stop()
{
    for (const auto& thd : _threads) {
        thd->stop();
    }
}
void scheduler_mt::wait()
{
    for (const auto& thd : _threads) {
        thd->wait();
    }
}
void scheduler_mt::run()
{
    for (const auto& thd : _threads) {
        thd->start();
    }
    for (const auto& thd : _threads) {
        thd->wait();
    }
}

} // namespace schedulers
} // namespace gr