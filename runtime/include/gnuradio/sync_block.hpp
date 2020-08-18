#ifndef INCLUDED_SYNC_BLOCK_HPP
#define INCLUDED_SYNC_BLOCK_HPP

#include <algorithm>
#include <limits>
#include <gnuradio/block.hpp>


namespace gr {
/**
 * @brief synchronous 1:1 input to output with history
 *
 */
class sync_block : public block
{
public:
    sync_block(const std::string& name);

    // ~sync_block() {};
    /**
     * @brief Performs checks on inputs and outputs before and after the call
     * to the derived block's work function
     *
     * The sync_block guarantees that the input and output buffers to the
     * work function of the derived block fit the constraints of the 1:1
     * sample input/output relationship
     *
     * 1. Check all inputs and outputs have the same number of items
     * 2. Fix all inputs and outputs to the absolute min across ports
     * 3. Call the work() function on the derived block
     * 4. Throw runtime_error if n_produced is not the same on every port
     * 5. Set n_consumed = n_produced for every input port
     *
     * @param work_input
     * @param work_output
     * @return work_return_code_t
     */
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
            if (firsttime) {
                n_produced = w.n_produced;
                firsttime = false;
            }
            if (n_produced != w.n_produced) {
                allsame = false;
                break;
            }
        }
        if (!allsame) {
            throw new std::runtime_error(
                "outputs for sync_block must produce same number of items");
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