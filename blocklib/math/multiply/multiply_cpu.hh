#pragma once

#include <gnuradio/math/multiply.hh>

namespace gr {
namespace math {

template <class T>
class multiply_cpu : public multiply<T>
{
public:
    multiply_cpu(const typename multiply<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

protected:
    size_t d_num_inputs;
    size_t d_vlen;
};


} // namespace math
} // namespace gr
