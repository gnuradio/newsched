#pragma once

#include <gnuradio/block.hpp>

namespace gr {
class block_context : public block_interface
{
    protected:
        block_sptr block_impl;
    public:
        block_context(block_sptr b) { block_impl = b; }
    /**
     * @brief Abstract method to call signal processing work from a derived block
     *
     * @param work_input Vector of block_work_input structs
     * @param work_output Vector of block_work_output structs
     * @return work_return_code_t
     */
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) { return block_impl->work(work_input, work_output); }
    /**
     * @brief Wrapper for work to perform special checks and take care of special
     * cases for certain types of blocks, e.g. sync_block, decim_block
     *
     * @param work_input Vector of block_work_input structs
     * @param work_output Vector of block_work_output structs
     * @return work_return_code_t
     */
    virtual work_return_code_t do_work(std::vector<block_work_input>& work_input,
                                       std::vector<block_work_output>& work_output) { return block_impl->do_work(work_input, work_output); }

    virtual void set_scheduler(std::shared_ptr<scheduler> sched) { block_impl->set_scheduler(sched); }
};

} // namespace gr