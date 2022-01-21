#include "delay_cpu.hh"
#include "delay_cpu_gen.hh"

namespace gr {
namespace blocks {

delay_cpu::delay_cpu(const block_args& args) : INHERITED_CONSTRUCTORS
{
    set_dly(args.dly);
}

void delay_cpu::set_dly(size_t d)
{
    // only set a new delta if there is a change in the delay; this
    // protects from quickly-repeated calls to this function that
    // would end with d_delta=0.
    if (d != dly()) {
        std::scoped_lock l(d_mutex);
        int old = dly();
        d_delay = d;
        d_delta += dly() - old;
    }
}

work_return_code_t delay_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                   std::vector<block_work_output_sptr>& work_output)
{
    std::scoped_lock l(d_mutex);
    assert(work_input.size() == work_output.size());
    auto itemsize = work_output[0]->buffer->item_size();

    const uint8_t* iptr;
    uint8_t* optr;
    int cons, ret;
    int noutput_items =
        std::min(work_output[0]->n_items, work_input[0]->n_items); // - (dly() - d_delta);

    // No change in delay; just memcpy ins to outs
    if (d_delta == 0) {
        for (size_t i = 0; i < work_input.size(); i++) {
            iptr = work_input[i]->items<uint8_t>();
            optr = work_output[i]->items<uint8_t>();
            std::memcpy(optr, iptr, noutput_items * itemsize);
        }
        cons = noutput_items;
        ret = noutput_items;
    }

    // Skip over d_delta items on the input
    else if (d_delta < 0) {
        int n_to_copy, n_adj;
        int delta = -d_delta;
        n_to_copy = std::max(0, noutput_items - delta);
        n_adj = std::min(delta, noutput_items);
        for (size_t i = 0; i < work_input.size(); i++) {
            iptr = work_input[i]->items<uint8_t>();
            optr = work_output[i]->items<uint8_t>();
            std::memcpy(optr, iptr + delta * itemsize, n_to_copy * itemsize);
        }
        cons = noutput_items;
        ret = n_to_copy;
        delta -= n_adj;
        d_delta = -delta;
    }

    // produce but not consume (inserts zeros)
    else { // d_delta > 0
        int n_from_input, n_padding;
        n_from_input = std::max(0, noutput_items - d_delta);
        n_padding = std::min(d_delta, noutput_items);
        for (size_t i = 0; i < work_input.size(); i++) {
            iptr = work_input[i]->items<uint8_t>();
            optr = work_output[i]->items<uint8_t>();
            std::memset(optr, 0, n_padding * itemsize);
            std::memcpy(optr + n_padding * itemsize, iptr, n_from_input * itemsize);
        }
        cons = n_from_input;
        ret = noutput_items;
        d_delta -= n_padding;
    }

    consume_each(cons, work_input);
    produce_each(ret, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr