#pragma once

#include <gnuradio/blocks/annotator.hh>

namespace gr {
namespace blocks {

class annotator_cpu : public annotator
{
public:
    annotator_cpu(const block_args& args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

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