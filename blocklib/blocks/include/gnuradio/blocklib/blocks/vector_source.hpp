#ifndef INCLUDED_VECTOR_SOURCE_HPP
#define INCLUDED_VECTOR_SOURCE_HPP

#include <gnuradio/sync_block.hpp>

namespace gr {
namespace blocks {

template <class T>
class vector_source : virtual public sync_block
{

public:
    enum params : uint32_t { id_data, id_repeat, id_vlen, id_tags, num_params };

    typedef std::shared_ptr<vector_source> sptr;
    static sptr make(const std::vector<T>& data,
                     bool repeat = false,
                     unsigned int vlen = 1,
                     const std::vector<tag_t>& tags = std::vector<tag_t>())
    {

        auto ptr = std::make_shared<vector_source>(vector_source(data, repeat, vlen, tags));


        ptr->add_port(port<T>::make("output",
                                    port_direction_t::OUTPUT,
                                    port_type_t::STREAM,
                                    std::vector<size_t>{ vlen }));

        ptr->add_param(
            param<std::vector<T>>::make(vector_source::params::id_data, "data", data, &ptr->_data));
        ptr->add_param(
            param<bool>::make(vector_source::params::id_repeat, "repeat", repeat, &ptr->_repeat));
        ptr->add_param(
            param<size_t>::make(vector_source::params::id_vlen, "vlen", vlen, &ptr->_vlen));
        ptr->add_param(
            param<std::vector<tag_t>>::make(vector_source::params::id_tags, "tags", tags, &ptr->_tags));

        return ptr;
    }

    vector_source(const std::vector<T>& data,
                                bool repeat,
                                unsigned int vlen,
                                const std::vector<tag_t>& tags);
    // ~vector_source() {};

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);

    void rewind(){};
    void set_data(const std::vector<T>& data,
                  const std::vector<tag_t>& tags = std::vector<tag_t>());
    void set_repeat(bool repeat);

private:
    std::vector<T> _data;
    bool _repeat;
    unsigned int _offset;
    size_t _vlen;
    bool _settags;
    std::vector<tag_t> _tags;
};

typedef vector_source<std::uint8_t> vector_source_b;
typedef vector_source<std::int16_t> vector_source_s;
typedef vector_source<std::int32_t> vector_source_i;
typedef vector_source<float> vector_source_f;
typedef vector_source<gr_complex> vector_source_c;

} // namespace blocks
} // namespace gr
#endif