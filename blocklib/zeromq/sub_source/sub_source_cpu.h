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

private:
    // private variables here
};

} // namespace zeromq
} // namespace gr