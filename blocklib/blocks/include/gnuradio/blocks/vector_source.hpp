#pragma once

#include <gnuradio/sync_block.hpp>

namespace gr {
namespace blocks {

template <class T>
class vector_source : virtual public sync_block
{
public:
    enum params : uint32_t { idd_data, idd_repeat, idd_vlen, idd_tags, num_params };

    typedef std::shared_ptr<vector_source> sptr;
    static sptr make(const std::vector<T>& data,
                     bool repeat = false,
                     unsigned int vlen = 1,
                     const std::vector<tag_t>& tags = std::vector<tag_t>())
    {

        auto ptr = std::make_shared<vector_source>(data, repeat, vlen, tags);

        ptr->add_port(port<T>::make("output",
                                    port_direction_t::OUTPUT,
                                    std::vector<size_t>{ vlen }));

        return ptr;
    }

    vector_source(const std::vector<T>& data,
                  bool repeat,
                  unsigned int vlen,
                  const std::vector<tag_t>& tags);

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);

private:
    std::vector<T> d_data;
    bool d_repeat;
    unsigned int d_offset;
    size_t d_vlen;
    bool d_settags;
    std::vector<tag_t> d_tags;
};

typedef vector_source<std::uint8_t> vector_source_b;
typedef vector_source<std::int16_t> vector_source_s;
typedef vector_source<std::int32_t> vector_source_i;
typedef vector_source<float> vector_source_f;
typedef vector_source<gr_complex> vector_source_c;

} // namespace blocks
} // namespace gr
