#pragma once

#include "block_impl.h"
#include <gnuradio/soapy/source.h>
namespace gr {
namespace soapy {

template <class T>
class source_cpu : public source<T>, public block_impl
{
public:
    source_cpu(const typename source<T>::block_args& args);

    work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;

private:
};


} // namespace soapy
} // namespace gr
