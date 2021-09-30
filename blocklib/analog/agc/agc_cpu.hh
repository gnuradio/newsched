#pragma once

#include <gnuradio/analog/agc.hh>
#include <gnuradio/analog/kernel/agc.hh>

namespace gr {
namespace analog {

template <class T>
class agc_cpu : public agc<T>, kernel::agc<T>
{
public:
    agc_cpu(const typename agc<T>::block_args& args)
        : sync_block("agc"), agc<T>(args), kernel::agc<T>(args.rate, args.reference, args.gain, 65536)
    {
    }
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
};

} // namespace analog
} // namespace gr
