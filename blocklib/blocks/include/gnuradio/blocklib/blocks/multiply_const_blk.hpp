#ifndef INCLUDED_MULTIPLY_CONST_HPP
#define INCLUDED_MULTIPLY_CONST_HPP

#include <gnuradio/blocklib/sync_block.hpp>

namespace gr {
namespace blocks {

template <class T>
class multiply_const : public sync_block
{

private:

    T d_k;
    const size_t d_vlen;

    void ports_and_params(size_t);

public:
    enum params : uint32_t { id_k, id_vlen, num_params };

    multiply_const(T k, size_t vlen = 1);
    // ~multiply_const() {};

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);


    virtual void on_parameter_change(param_action_base param) override;
    virtual void on_parameter_query(param_action_base& param) override;
    std::any handle_do_a_bunch_of_things(std::vector<std::any> args)
    {
        auto xi = std::any_cast<int>(args[0]);
        auto yi = std::any_cast<double>(args[1]);
        auto zi = std::any_cast<std::vector<gr_complex>>(args[2]);

        return std::make_any<double>(xi*yi);
    }

    // These methods should be automatically generated
    // setters/getters/callback wrappers
    double do_a_bunch_of_things(const int x, const double y, const std::vector<gr_complex>& z);
    void set_k(T k);
    T k();



};

typedef multiply_const<std::int16_t> multiply_const_ss;
typedef multiply_const<std::int32_t> multiply_const_ii;
typedef multiply_const<float> multiply_const_ff;
typedef multiply_const<gr_complex> multiply_const_cc;

} // namespace blocks
} // namespace gr
#endif