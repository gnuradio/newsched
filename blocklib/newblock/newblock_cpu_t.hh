#pragma once

#include <gnuradio/newmod/newblock.hh>

namespace gr {
namespace newmod {

#define PARAM_LIST T k, size_t vlen
#define PARAM_VALS k, vlen

template <class T>
class newblock_cpu : public newblock<T>
{
public:
    newblock_cpu(const typename newblock<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    T d_k;
    size_t d_vlen;
};


} // namespace newmod
} // namespace gr
