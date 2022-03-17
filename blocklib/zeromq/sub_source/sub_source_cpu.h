#pragma once

#include "base.h"
#include <gnuradio/zeromq/sub_source.h>

namespace gr {
namespace zeromq {

class sub_source_cpu : public virtual sub_source, public virtual base_source
{
public:
    sub_source_cpu(block_args args);
    work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;

    // Since vsize can be set as 0, then inferred on flowgraph init, set it during start()
    bool start() override
    {
        set_vsize(this->output_stream_ports()[0]->itemsize());
        return sub_source::start();
    }

private:
    // private variables here
};

} // namespace zeromq
} // namespace gr