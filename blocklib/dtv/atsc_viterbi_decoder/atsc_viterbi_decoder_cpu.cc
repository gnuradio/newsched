#include "atsc_viterbi_decoder_cpu.hh"

#include <gnuradio/dtv/atsc_plinfo.hh>

namespace gr {
namespace dtv {

atsc_viterbi_decoder::sptr atsc_viterbi_decoder::make_cpu(const block_args& args) { return std::make_shared<atsc_viterbi_decoder_cpu>(args); }

atsc_viterbi_decoder_cpu::atsc_viterbi_decoder_cpu(const block_args& args) : atsc_viterbi_decoder(args)
{
    set_output_multiple(NCODERS);  // TODO - how to handle

    /*
     * These fifo's handle the alignment problem caused by the
     * inherent decoding delay of the individual viterbi decoders.
     * The net result is that this entire block has a pipeline latency
     * of 12 complete segments.
     *
     * If anybody cares, it is possible to do it with less delay, but
     * this approach is at least somewhat understandable...
     */

    // the -4 is for the 4 sync symbols
    const int fifo_size = ATSC_DATA_SEGMENT_LENGTH - 4 - viterbi[0].delay();
    fifo.reserve(NCODERS);
    for (int i = 0; i < NCODERS; i++)
        fifo.emplace_back(fifo_size);

    reset();

}

void atsc_viterbi_decoder_cpu::reset()
{
    for (int i = 0; i < NCODERS; i++)
        fifo[i].reset();
}

std::vector<float> atsc_viterbi_decoder_cpu::decoder_metrics() const
{
    std::vector<float> metrics(NCODERS);
    for (int i = 0; i < NCODERS; i++)
        metrics[i] = viterbi[i].best_state_metric();
    return metrics;
}


work_return_code_t atsc_viterbi_decoder_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
   auto in = static_cast<const float*>(work_input[0].items());
    auto out = static_cast<uint8_t*>(work_output[0].items());
    auto plin = static_cast<const plinfo*>(work_input[1].items());
    auto plout = static_cast<plinfo*>(work_output[1].items());

    auto noutput_items = work_output[0].n_items;
    // The way the fs_checker works ensures we start getting packets
    // starting with a field sync, and our input multiple is set to
    // 12, so we should always get a mod 12 numbered first packet
    assert(noutput_items % NCODERS == 0);

    int dbwhere;
    int dbindex;
    int shift;
    float symbols[NCODERS][enco_which_max];
    unsigned char dibits[NCODERS][enco_which_max];

    unsigned char out_copy[OUTPUT_SIZE];

    for (int i = 0; i < noutput_items; i += NCODERS) {

        /* Build a continuous symbol buffer for each encoder */
        for (unsigned int encoder = 0; encoder < NCODERS; encoder++)
            for (unsigned int k = 0; k < enco_which_max; k++)
                symbols[encoder][k] =
                    in[(i + (enco_which_syms[encoder][k] / ATSC_DATA_SEGMENT_LENGTH)) *
                           ATSC_DATA_SEGMENT_LENGTH +
                       enco_which_syms[encoder][k] % ATSC_DATA_SEGMENT_LENGTH];


        /* Now run each of the 12 Viterbi decoders over their subset of
           the input symbols */
        for (unsigned int encoder = 0; encoder < NCODERS; encoder++)
            for (unsigned int k = 0; k < enco_which_max; k++)
                dibits[encoder][k] = viterbi[encoder].decode(symbols[encoder][k]);

        /* Move dibits into their location in the output buffer */
        for (unsigned int encoder = 0; encoder < NCODERS; encoder++) {
            for (unsigned int k = 0; k < enco_which_max; k++) {
                /* Store the dibit into the output data segment */
                dbwhere = enco_which_dibits[encoder][k];
                dbindex = dbwhere >> 3;
                shift = dbwhere & 0x7;
                out_copy[dbindex] = (out_copy[dbindex] & ~(0x03 << shift)) |
                                    (fifo[encoder].stuff(dibits[encoder][k]) << shift);
            } /* Symbols fed into one encoder */
        }     /* Encoders */

        // copy output from contiguous temp buffer into final output
        for (int j = 0; j < NCODERS; j++) {

            memcpy(&out[(i + j) * ATSC_MPEG_RS_ENCODED_LENGTH],
                   &out_copy[j * ATSC_MPEG_RS_ENCODED_LENGTH],
                   ATSC_MPEG_RS_ENCODED_LENGTH * sizeof(out_copy[0]));

            plinfo::delay(plout[i + j], plin[i + j], NCODERS);
        }
    }

    produce_each(noutput_items,work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace dtv
} // namespace gr