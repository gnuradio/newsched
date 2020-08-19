#ifndef INCLUDED_MULTIPLY_CONST_HPP
#define INCLUDED_MULTIPLY_CONST_HPP

#include <gnuradio/sync_block.hpp>

namespace gr {
namespace blocks {

template <class T>
class multiply_const : public sync_block
{
public:
    enum params : uint32_t { id_k, id_vlen, num_params };

    typedef std::shared_ptr<multiply_const> sptr;
    static sptr make(const T k, const size_t vlen = 1)
    {
        auto ptr = std::make_shared<multiply_const>(multiply_const<T>());

        ptr->add_port(port<T>::make("input",
                                    port_direction_t::INPUT,
                                    port_type_t::STREAM,
                                    std::vector<size_t>{ vlen }));
        ptr->add_port(port<T>::make("output",
                                    port_direction_t::OUTPUT,
                                    port_type_t::STREAM,
                                    std::vector<size_t>{ vlen }));

        ptr->add_param(
            param<T>::make(multiply_const<T>::params::id_k, "k", k, &ptr->d_k));

        // TODO: vlen should be const and unchangeable as a parameter
        ptr->add_param(param<size_t>::make(
            multiply_const<T>::params::id_vlen, "vlen", vlen, &ptr->d_vlen));

        ptr->register_callback("do_a_bunch_of_things", [ptr](auto args) {
            return ptr->handle_do_a_bunch_of_things(args); });

        return ptr;

    }
    multiply_const();
    // ~multiply_const() {};

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);


    std::any handle_do_a_bunch_of_things(std::vector<std::any> args)
    {
            auto xi = std::any_cast<int>(args[0]);
            auto yi = std::any_cast<double>(args[1]);
            auto zi = std::any_cast<std::vector<gr_complex>>(args[2]);

            return std::make_any<double>(xi * yi);
    }

    // These methods should be automatically generated or macros
    // setters/getters/callback wrappers
    double do_a_bunch_of_things(const int x, const double y, const std::vector<gr_complex>& z);
    void set_k(T k) {return request_parameter_change<T>(params::id_k, k);}
    T k() { return request_parameter_query<T>(params::id_k); }

private:

    T d_k;
    size_t d_vlen;

    };

    typedef multiply_const<std::int16_t> multiply_const_ss;
    typedef multiply_const<std::int32_t> multiply_const_ii;
    typedef multiply_const<float> multiply_const_ff;
    typedef multiply_const<gr_complex> multiply_const_cc;

} // namespace blocks
} // namespace blocks
#endif