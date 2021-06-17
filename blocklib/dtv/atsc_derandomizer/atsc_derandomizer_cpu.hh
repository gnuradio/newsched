#pragma once

#include <gnuradio/dtv/atsc_derandomizer.hh>

#include "atsc_randomize.hh"

namespace gr {
namespace dtv {

class atsc_derandomizer_cpu : public atsc_derandomizer
{
public:
    atsc_derandomizer_cpu(const block_args& args);
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

private:
    atsc_randomize d_rand;
};

} // namespace dtv
} // namespace gr