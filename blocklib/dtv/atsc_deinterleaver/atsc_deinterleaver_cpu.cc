#include "atsc_deinterleaver_cpu.hh"
#include <gnuradio/dtv/atsc_consts.hh>
#include <gnuradio/dtv/atsc_plinfo.hh>

namespace gr {
namespace dtv {

atsc_deinterleaver::sptr atsc_deinterleaver::make_cpu(const block_args& args) { return std::make_shared<atsc_deinterleaver_cpu>(args); }

atsc_deinterleaver_cpu::atsc_deinterleaver_cpu(const block_args& args) : atsc_deinterleaver(args), alignment_fifo(156)
{
     m_fifo.reserve(s_interleavers);

    for (int i = 0; i < s_interleavers; i++)
        m_fifo.emplace_back((s_interleavers - 1 - i) * 4);

    sync();

    set_tag_propagation_policy(tag_propagation_policy_t::TPP_CUSTOM);
}

void atsc_deinterleaver_cpu::reset()
{
    sync();

    for (auto& i : m_fifo)
        i.reset();
}


work_return_code_t atsc_deinterleaver_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const uint8_t*>(work_input[0].items());
    auto out = static_cast<uint8_t*>(work_output[0].items());
    auto plin = static_cast<const plinfo*>(work_input[1].items());
    auto plout = static_cast<plinfo*>(work_output[1].items());
    auto noutput_items = work_output[0].n_items;

    for (int i = 0; i < noutput_items; i++) {
        assert(plin[i].regular_seg_p());

        // reset commutator if required using INPUT pipeline info
        if (plin[i].first_regular_seg_p())
            sync();

        // remap OUTPUT pipeline info to reflect all data segment end-to-end delay
        plinfo::delay(plout[i], plin[i], s_interleavers);

        // now do the actual deinterleaving
        for (unsigned int j = 0; j < ATSC_MPEG_RS_ENCODED_LENGTH; j++) {
            out[i * ATSC_MPEG_RS_ENCODED_LENGTH + j] =
                alignment_fifo.stuff(transform(in[i * ATSC_MPEG_RS_ENCODED_LENGTH + j]));
        }
    }

    produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace dtv
} // namespace gr