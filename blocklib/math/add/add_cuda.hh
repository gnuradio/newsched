#pragma once

#include <gnuradio/math/add.hh>
#include <cusp/add.cuh>

namespace gr {
namespace math {

template <class T>
class add_cuda : public add<T>
{
public:
    add_cuda(const typename add<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

private:
    const size_t d_vlen;
    const size_t d_nports;
    std::vector<const void *> d_in_items;

    std::shared_ptr<cusp::add<T>> p_add_kernel;
};


} // namespace math
} // namespace gr
