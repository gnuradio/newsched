#pragma once

#include <gnuradio/sync_block.hpp>

#include <cstdint>
#include <mutex>
namespace gr {
namespace blocks {

template <class T>
class vector_sink : virtual public sync_block
{
public:
    typedef std::shared_ptr<vector_sink> sptr;

    static sptr make(const size_t vlen = 1, const size_t reserve_items = 1024)
    {
        auto ptr = std::make_shared<vector_sink>(vlen, reserve_items);

        ptr->add_port(port<T>::make("input",
                                    port_direction_t::INPUT,
                                    std::vector<size_t>{ vlen }));

        return ptr;
    }

    vector_sink(const size_t vlen = 1, const size_t reserve_items = 1024);

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output);
    
    std::vector<T> data()
    {
        return d_data;
    }

private:
    std::vector<T> d_data;
    std::vector<tag_t> d_tags;
    size_t d_vlen;
};
typedef vector_sink<std::uint8_t> vector_sink_b;
typedef vector_sink<std::int16_t> vector_sink_s;
typedef vector_sink<std::int32_t> vector_sink_i;
typedef vector_sink<float> vector_sink_f;
typedef vector_sink<gr_complex> vector_sink_c;
} /* namespace blocks */
} /* namespace gr */
