#pragma once

#include <gnuradio/blocks/vector_source.hh>

namespace gr {
namespace blocks {

template <class T>
class vector_source_cpu : public vector_source<T>
{
public:
    vector_source_cpu(const typename vector_source<T>::block_args& args);

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;


protected:
    std::vector<T> d_data;
    bool d_repeat;
    unsigned int d_offset;
    size_t d_vlen;
    bool d_settags;
    std::vector<tag_t> d_tags;
};


} // namespace blocks
} // namespace gr
