#pragma once

#include <gnuradio/sync_block.hpp>

namespace gr {
namespace blocks {

template <class T>
class multiply_const : public sync_block
{
public:
    typedef std::shared_ptr<multiply_const> sptr;
    static sptr make(const T k, const size_t vlen = 1)
    {
        auto ptr = std::make_shared<multiply_const>(multiply_const<T>(k,vlen));

        ptr->add_port(port<T>::make("input",
                                    port_direction_t::INPUT,
                                    std::vector<size_t>{ vlen }));
        ptr->add_port(port<T>::make("output",
                                    port_direction_t::OUTPUT,
                                    std::vector<size_t>{ vlen }));

        return ptr;

    }
    multiply_const(T k, size_t vlen);

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);

private:
    T d_k;
    size_t d_vlen;
};

typedef multiply_const<std::int16_t> multiply_const_ss;
typedef multiply_const<std::int32_t> multiply_const_ii;
typedef multiply_const<float> multiply_const_ff;
typedef multiply_const<gr_complex> multiply_const_cc;

} // namespace blocks
} // namespace gr
