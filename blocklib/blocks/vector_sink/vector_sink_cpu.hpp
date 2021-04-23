#pragma once

#include <gnuradio/blocks/vector_sink.hpp>

namespace gr {
namespace blocks {

#define PARAM_LIST const size_t vlen, const size_t reserve_items
#define PARAM_VALS vlen, reserve_items

template <class T>
class vector_sink_cpu : public vector_sink<T>
{
public:
    vector_sink_cpu(const size_t vlen, const size_t reserve_items);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

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
