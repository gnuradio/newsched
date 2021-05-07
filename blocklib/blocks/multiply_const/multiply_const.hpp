#pragma once
#include <gnuradio/sync_block.hpp>
#include <gnuradio/types.hpp>

namespace gr {
namespace blocks {

template <class T>
class multiply_const : public sync_block
{
public:
    typedef struct {
        T k;
        size_t vlen = 1;
    } block_args;

    typedef std::shared_ptr<multiply_const> sptr;
    multiply_const(const block_args& args) : sync_block("multiply_const")
    {
        add_port(
            port<T>::make("in", port_direction_t::INPUT, std::vector<size_t>{ args.vlen }));

        add_port(
            port<T>::make("out", port_direction_t::OUTPUT, std::vector<size_t>{ args.vlen }));
    }

    /**
     * @brief Set the implementation to CPU and return a shared pointer to the block
     * instance
     *
     * @return std::shared_ptr<multiply_const>
     */
    static sptr make_cpu(const block_args& args);

    /**
     * @brief Set the implementation to CUDA and return a shared pointer to the block
     * instance
     *
     * @return std::shared_ptr<multiply_const>
     */
    static sptr make_cuda(const block_args& args);
};

typedef multiply_const<int16_t> multiply_const_ss;
typedef multiply_const<int32_t> multiply_const_ii;
typedef multiply_const<float> multiply_const_ff;
typedef multiply_const<gr_complex> multiply_const_cc;

} // namespace blocks
} // namespace gr