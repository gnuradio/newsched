#pragma once

#include <gnuradio/block_work_io.hpp>
#include <gnuradio/node.hpp>
#include <gnuradio/gpdict.hpp>

namespace gr {

class scheduler; // Forward declaration to scheduler class

class block_interface
{

public:
    virtual bool start() = 0;
    virtual bool stop() = 0;

    /**
     * @brief Abstract method to call signal processing work from a derived block
     *
     * @param work_input Vector of block_work_input structs
     * @param work_output Vector of block_work_output structs
     * @return work_return_code_t
     */
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) = 0;
    /**
     * @brief Wrapper for work to perform special checks and take care of special
     * cases for certain types of blocks, e.g. sync_block, decim_block
     *
     * @param work_input Vector of block_work_input structs
     * @param work_output Vector of block_work_output structs
     * @return work_return_code_t
     */
    virtual work_return_code_t do_work(std::vector<block_work_input>& work_input,
                                       std::vector<block_work_output>& work_output) = 0;

    virtual void set_scheduler(std::shared_ptr<scheduler> sched) = 0;
};


} // namespace gr
