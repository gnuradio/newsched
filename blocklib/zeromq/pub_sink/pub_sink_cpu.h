#pragma once

#include "base.h"
#include <gnuradio/zeromq/pub_sink.h>

namespace gr {
namespace zeromq {

class pub_sink_cpu : public virtual pub_sink, public virtual base_sink
{
public:
    pub_sink_cpu(block_args args);
    work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                            std::vector<block_work_output_sptr>& work_output) override;
    std::string last_endpoint() const override { return base_sink::last_endpoint(); }

    // Since vsize can be set as 0, then inferred on flowgraph init, set it during start()
    bool start() override
    {
        set_vsize(this->input_stream_ports()[0]->itemsize());
        return pub_sink::start();
    }

private:
    // private variables here
};

} // namespace zeromq
} // namespace gr