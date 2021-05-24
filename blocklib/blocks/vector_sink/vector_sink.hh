#pragma once

#include <gnuradio/sync_block.hh>
#include <gnuradio/types.hh>

namespace gr {
namespace blocks {

template <class T>
class vector_sink : public sync_block
{
public:
    typedef std::shared_ptr<vector_sink> sptr;

    struct block_args {
        size_t vlen = 1;
        size_t reserve_items = 1024;
    };

    vector_sink(const block_args& args) : sync_block("vector_sink")
    {
        add_port(
            port<T>::make("input", port_direction_t::INPUT, std::vector<size_t>{ args.vlen }));
    }

    virtual std::vector<T> data() = 0;



    enum class available_impl {
        CPU,
        // CUDA
    };
    static sptr make(block_args args = {}, available_impl impl = available_impl::CPU)
    {
        switch (impl) {
        case available_impl::CPU:
            return make_cpu(args);
            break;
        // case available_impl::CUDA:
        //     return make_cuda(args);
        //     break;
        default:
            throw std::invalid_argument(
                "blocks::copy - invalid implementation specified");
        }
    }


    /**
     * @brief Set the implementation to CPU and return a shared pointer to the block
     * instance
     *
     * @return std::shared_ptr<vector_sink>
     */
    static sptr make_cpu(const block_args& args = {});
};

typedef vector_sink<std::uint8_t> vector_sink_b;
typedef vector_sink<std::int16_t> vector_sink_s;
typedef vector_sink<std::int32_t> vector_sink_i;
typedef vector_sink<float> vector_sink_f;
typedef vector_sink<gr_complex> vector_sink_c;

} // namespace blocks
} // namespace gr
