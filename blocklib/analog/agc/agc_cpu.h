#pragma once

#include <gnuradio/analog/agc.h>
#include <gnuradio/kernel/analog/agc.h>

namespace gr {
namespace analog {

template <class T>
class agc_cpu : public agc<T>, kernel::analog::agc<T>
{
public:
    agc_cpu(const typename agc<T>::block_args& args);
    work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;

protected:
};

} // namespace analog
} // namespace gr
