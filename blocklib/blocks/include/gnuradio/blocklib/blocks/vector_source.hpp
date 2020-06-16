#ifndef INCLUDED_VECTOR_SOURCE_HPP
#define INCLUDED_VECTOR_SOURCE_HPP

#include <gnuradio/blocklib/sync_block.hpp>

namespace gr {
namespace blocks {

template <class T>
class vector_source : virtual public sync_block
{
private:
    // static const io_signature_capability d_input_signature_capability =
    // io_signature_capability(0, 0); static const io_signature_capability
    // d_output_signature_capability = io_signature_capability(1, -1);

    std::vector<T> d_data;
    bool d_repeat;
    unsigned int d_offset;
    unsigned int d_vlen;
    bool d_settags;
    std::vector<tag_t> d_tags;

public:
    enum params : uint32_t { data, repeat, vlen, num_params };
    vector_source(const std::vector<T>& data,
                  bool repeat = false,
                  unsigned int vlen = 1,
                  const std::vector<tag_t>& tags = std::vector<tag_t>());
    // ~vector_source() {};

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);

    void rewind(){};
    void set_data(const std::vector<T>& data,
                  const std::vector<tag_t>& tags = std::vector<tag_t>());
    void set_repeat(bool repeat);
};

typedef vector_source<std::uint8_t> vector_source_b;
typedef vector_source<std::int16_t> vector_source_s;
typedef vector_source<std::int32_t> vector_source_i;
typedef vector_source<float> vector_source_f;
typedef vector_source<gr_complex> vector_source_c;

} // namespace blocks
} // namespace gr
#endif