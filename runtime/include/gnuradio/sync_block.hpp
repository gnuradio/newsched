#ifndef INCLUDED_SYNC_BLOCK_HPP
#define INCLUDED_SYNC_BLOCK_HPP

#include <gnuradio/block.hpp>
#include <algorithm>
#include <limits>

namespace gr {
class sync_block : public block
{
public:
    sync_block(const std::string& name,
               const io_signature& input_signature,
               const io_signature& output_signature);
    // ~sync_block();
    work_return_code_t do_work(std::vector<block_work_input>& work_input,
                               std::vector<block_work_output>& work_output)
    {
        // Check all inputs and outputs have the same number of items
        int min_num_items = std::numeric_limits<int>::max();
        for (auto& w : work_input) {
            min_num_items = std::min(min_num_items, w.n_items);
        }
        for (auto& w : work_output) {
            min_num_items = std::min(min_num_items, w.n_items);
        }

        // all inputs and outputs need to be fixed to the absolute min
        for (auto& w : work_input) {
            w.n_items = min_num_items;
        }
        for (auto& w : work_output) {
            w.n_items = min_num_items;
        }

        work_return_code_t ret = work(work_input, work_output);

        // For a sync block the n_produced must be the same on every
        // output port

        bool firsttime = true;
        int n_produced = -1;
        bool allsame = true;
        for (auto& w : work_output) {
            if (firsttime)
            {
                n_produced = w.n_produced;
                firsttime = false;
            }
            if (n_produced != w.n_produced)
            {
                allsame = false;
                break;
            }
        }
        if (!allsame)
        {
            throw new std::runtime_error("outputs for sync_block must produce same number of items");
        }

        // by definition of a sync block the n_consumed must be equal to n_produced
        // also, a sync block must consume all of its items
        for (auto& w : work_input) {
            w.n_consumed = n_produced < 0 ? w.n_items : n_produced;
        }

        return ret;
    };
};
} // namespace gr
#endif