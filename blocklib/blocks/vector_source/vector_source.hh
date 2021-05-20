#pragma once

#include <gnuradio/sync_block.hh>
#include <gnuradio/types.hh>

namespace gr {
namespace blocks {

template <class T>
class vector_source : public sync_block
{
public:
    typedef std::shared_ptr<vector_source> sptr;
    vector_source(unsigned int vlen) : sync_block("vector_source")
    {
        add_port(port<T>::make(
            "output", port_direction_t::OUTPUT, std::vector<size_t>{ vlen }));
    }


    /**
     * @brief Set the implementation to CPU and return a shared pointer to the block
     * instance
     *
     * @return std::shared_ptr<vector_source>
     */
    static sptr make_cpu(const std::vector<T>& data,
                    bool repeat = false,
                    unsigned int vlen = 1,
                    const std::vector<tag_t>& tags = std::vector<tag_t>());
};

typedef vector_source<std::uint8_t> vector_source_b;
typedef vector_source<std::int16_t> vector_source_s;
typedef vector_source<std::int32_t> vector_source_i;
typedef vector_source<float> vector_source_f;
typedef vector_source<gr_complex> vector_source_c;

} // namespace blocks
} // namespace gr