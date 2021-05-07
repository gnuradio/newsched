#pragma once

#include <gnuradio/blocks/multiply_const.hpp>

namespace gr {
namespace blocks {

#define PARAM_LIST T k, size_t vlen
#define PARAM_VALS k, vlen

template <class T>
class multiply_const_cpu : public multiply_const<T>
{
public:
    multiply_const_cpu(const typename multiply_const<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    T d_k;
    size_t d_vlen;
};


} // namespace blocks
} // namespace gr
