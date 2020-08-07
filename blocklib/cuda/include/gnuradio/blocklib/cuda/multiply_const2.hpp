#pragma once

#include <gnuradio/blocklib/sync_block.hpp>

namespace gr {
namespace cuda {
template <class T>
class multiply_const2 : public sync_block
{
public:
    enum params : uint32_t { id_k, id_vlen, num_params };

    typedef std::shared_ptr<multiply_const2> sptr;
    static sptr make(const T k, const size_t vlen = 1)
    {
        auto ptr = std::make_shared<multiply_const2>(multiply_const2<T>());

        ptr->add_port(port<T>::make("input",
                                    port_direction_t::INPUT,
                                    port_type_t::STREAM,
                                    std::vector<size_t>{ vlen }));
        ptr->add_port(port<T>::make("output",
                                    port_direction_t::OUTPUT,
                                    port_type_t::STREAM,
                                    std::vector<size_t>{ vlen }));

        ptr->add_param(
            param<T>::make(multiply_const2<T>::params::id_k, "k", k, &ptr->d_k));

        // TODO: vlen should be const and unchangeable as a parameter
        ptr->add_param(param<size_t>::make(
            multiply_const2<T>::params::id_vlen, "vlen", vlen, &ptr->d_vlen));

        return ptr;

    }
    multiply_const2()
    : sync_block("multiply_const2 (cuda)") {}
    // ~multiply_const2() {};

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);

    // These methods should be automatically generated or macros
    // setters/getters/callback wrappers
    void set_k(T k) {return request_parameter_change<T>(params::id_k, k);}
    T k() { return request_parameter_query<T>(params::id_k); }

private:

    T d_k;
    size_t d_vlen;

    };

    typedef multiply_const2<float> multiply_const2_ff;

} // namespace blocks
} // namespace blocks