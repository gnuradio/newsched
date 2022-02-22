#include "agc_cpu.h"
#include "agc_cpu_gen.h"

namespace gr {
namespace analog {

template <class T>
agc_cpu<T>::agc_cpu(const typename agc<T>::block_args& args)
    : INHERITED_CONSTRUCTORS(T), kernel::agc<T>(args.rate, args.reference, args.gain, 65536)
{
}

template <class T>
work_return_code_t agc_cpu<T>::work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0]->items<T>();
    auto out = work_output[0]->items<T>();
    auto noutput_items = work_output[0]->n_items;
    kernel::agc<T>::scaleN(out, in, noutput_items);

    work_output[0]->n_produced = noutput_items;
    return work_return_code_t::WORK_OK;
}

} // namespace analog
} // namespace gr

