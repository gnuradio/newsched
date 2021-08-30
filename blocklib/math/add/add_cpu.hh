#pragma once

#include <gnuradio/math/add.hh>

namespace gr {
namespace math {

template <class T>
class add_cpu : public add<T>
{
public:
    add_cpu(const typename add<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

private:
    const size_t d_vlen;
    const size_t d_nports;
};


} // namespace math
} // namespace gr
