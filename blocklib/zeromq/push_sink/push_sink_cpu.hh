#pragma once

#include <gnuradio/zeromq/push_sink.hh>
#include "base.h"

namespace gr {
namespace zeromq {

class push_sink_cpu : public virtual push_sink, public virtual base_sink
{
public:
    push_sink_cpu(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;
    std::string last_endpoint () const { return base_sink::last_endpoint(); }
};

} // namespace blocks
} // namespace gr