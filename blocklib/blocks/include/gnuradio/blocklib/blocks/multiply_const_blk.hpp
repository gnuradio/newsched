#ifndef INCLUDED_MULTIPLY_CONST_HPP
#define INCLUDED_MULTIPLY_CONST_HPP

#include <gnuradio/blocklib/sync_block.hpp>

namespace gr {
namespace blocks {

template <class T>
class multiply_const : public sync_block
{
    T d_k;
    const size_t d_vlen;


public:
    enum params : uint32_t { id_k, id_vlen, num_params };

    multiply_const(T k, size_t vlen = 1);
    // ~multiply_const() {};

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);

    // const T k();
    // void set_k(T k);

    virtual void on_parameter_change(std::vector<param_change_base> params) override;
    // virtual std::any on_parameter_query(uint32_t id) override;

    // These methods should be automatically generated
    // setters/getters
    T k()
    {

        // // call back to the scheduler if ptr is not null
        // if (p_scheduler)
        // {
        //     //p_scheduler->request_parameter_value(alias(),)

        // }
        // // else go ahead and return parameter value
        // else
        // {
        //     return d_k; 
        // }

        return d_k;
        
    }

    void set_k(T k)
    {
        // // call back to the scheduler if ptr is not null
        // if (p_scheduler)
        // {
        //     // p_scheduler->request_parameter_change(alias(),)
        // }
        // // else go ahead and update parameter value
        // else
        // {
        //     on_parameter_change(std::vector<param_change_base>{param_change<T>(params::id_k, k, 0) });
        // }

        d_k = k;
    }
};

typedef multiply_const<std::int16_t> multiply_const_ss;
typedef multiply_const<std::int32_t> multiply_const_ii;
typedef multiply_const<float> multiply_const_ff;
typedef multiply_const<gr_complex> multiply_const_cc;

} // namespace blocks
} // namespace gr
#endif