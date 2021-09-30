#pragma once

#include <gnuradio/soapy/source.hh>
#include "block_impl.h"
namespace gr {
namespace soapy {

template <class T>
class source_cpu : public source<T>, public block_impl
{
public:
    source_cpu(const typename source<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

private:

};


} // namespace soapy
} // namespace gr
