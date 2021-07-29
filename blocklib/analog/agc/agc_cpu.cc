#include "agc_cpu.hh"

namespace gr {
namespace analog {

template <class T>
typename agc<T>::sptr agc<T>::make_cpu(const block_args& args)
{
    return std::make_shared<agc_cpu<T>>(args);
}

template <class T>
work_return_code_t agc_cpu<T>::work(std::vector<block_work_input>& work_input,
                                 std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const T*>(work_input[0].items());
    auto out = static_cast<T*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;
    kernel::agc<T>::scaleN(out, in, noutput_items);

    work_output[0].n_produced = noutput_items;
    return work_return_code_t::WORK_OK;
}

template class agc<float>;
template class agc<gr_complex>;

} // namespace analog
} // namespace gr