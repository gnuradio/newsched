#pragma once

#include <gnuradio/sync_block.hpp>
#include <gnuradio/types.hpp>

namespace gr {
namespace blocks {

template <class T>
class vector_sink : public sync_block
{
public:
    typedef std::shared_ptr<vector_sink> sptr;
    vector_sink(const size_t vlen, const size_t reserve_items) : sync_block("vector_sink")
    {
        add_port(
            port<T>::make("input", port_direction_t::INPUT, std::vector<size_t>{ vlen }));
    }

    virtual std::vector<T> data() = 0;

    /**
     * @brief Set the implementation to CPU and return a shared pointer to the block
     * instance
     *
     * @return std::shared_ptr<vector_sink>
     */
    static sptr cpu(const size_t vlen = 1, const size_t reserve_items = 1024);
};

typedef vector_sink<std::uint8_t> vector_sink_b;
typedef vector_sink<std::int16_t> vector_sink_s;
typedef vector_sink<std::int32_t> vector_sink_i;
typedef vector_sink<float> vector_sink_f;
typedef vector_sink<gr_complex> vector_sink_c;

} // namespace blocks
} // namespace gr