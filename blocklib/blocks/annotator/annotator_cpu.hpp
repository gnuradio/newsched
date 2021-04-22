#pragma once

#include <gnuradio/blocks/annotator.hpp>

namespace gr {
namespace blocks {

class annotator_cpu : public annotator
{
public:
    annotator_cpu(uint64_t when,
                  size_t itemsize,
                  size_t num_inputs,
                  size_t num_outputs,
                  tag_propagation_policy_t tpp);
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

    virtual std::vector<tag_t> data() const override { return d_stored_tags; };

private:
    const uint64_t d_when;
    uint64_t d_tag_counter;
    std::vector<tag_t> d_stored_tags;
    size_t d_num_inputs, d_num_outputs;
    tag_propagation_policy_t d_tpp;
};

} // namespace blocks
} // namespace gr