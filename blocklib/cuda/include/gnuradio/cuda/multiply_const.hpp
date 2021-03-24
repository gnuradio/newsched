#pragma once

#include <gnuradio/sync_block.hpp>

namespace gr {
namespace cuda {
template <class T>
class multiply_const : public sync_block
{
public:
    enum params : uint32_t { id_k, id_vlen, num_params };

    typedef std::shared_ptr<multiply_const> sptr;
    static sptr make(const T k, const size_t vlen = 1)
    {
        auto ptr = std::make_shared<multiply_const>();

        ptr->add_port(port<T>::make("input",
                                    port_direction_t::INPUT,
                                    std::vector<size_t>{ vlen }));
        ptr->add_port(port<T>::make("output",
                                    port_direction_t::OUTPUT,
                                    std::vector<size_t>{ vlen }));

        return ptr;

    }
    multiply_const()
    : sync_block("multiply_const (cuda)") {}
    // ~multiply_const() {};

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);


private:

    T d_k;
    size_t d_vlen;

    };

    typedef multiply_const<float> multiply_const_ff;

} // namespace blocks
} // namespace blocks
