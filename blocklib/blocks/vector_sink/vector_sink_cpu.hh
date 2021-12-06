#pragma once

#include <gnuradio/blocks/vector_sink.hh>

namespace gr {
namespace blocks {

template <class T>
class vector_sink_cpu : public vector_sink<T>
{
public:
    vector_sink_cpu(const typename vector_sink<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

    std::vector<T> data()
    {
        return d_data;
    }

protected:
    std::vector<T> d_data;
    std::vector<tag_t> d_tags;
    size_t d_vlen;
};


} // namespace blocks
} // namespace gr
