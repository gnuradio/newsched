#pragma once

#include <gnuradio/blocks/msg_forward.hh>

namespace gr {
namespace blocks {

class msg_forward_cpu : public msg_forward
{
public:
    msg_forward_cpu(block_args args) : msg_forward(args), d_itemsize(args.itemsize) {}
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t d_itemsize;
    void handle_msg_in(pmtf::pmt_sptr msg) { gr_log_info(_logger, "got message: "); }
};

} // namespace blocks
} // namespace gr