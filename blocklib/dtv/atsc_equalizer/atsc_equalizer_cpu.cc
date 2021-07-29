#include "atsc_equalizer_cpu.hh"

#include "atsc_pnXXX.hh"
#include "atsc_syminfo.hh"
#include "atsc_types.hh"
#include <gnuradio/dtv/atsc_consts.hh>
#include <gnuradio/dtv/atsc_equalizer.hh>
#include <volk/volk.h>

namespace gr {
namespace dtv {

atsc_equalizer::sptr atsc_equalizer::make_cpu(const block_args& args) { return std::make_shared<atsc_equalizer_cpu>(args); }

static float bin_map(int bit) { return bit ? +5 : -5; }

static void init_field_sync_common(float* p, int mask)
{
    int i = 0;

    p[i++] = bin_map(1); // data segment sync pulse
    p[i++] = bin_map(0);
    p[i++] = bin_map(0);
    p[i++] = bin_map(1);

    for (int j = 0; j < 511; j++) // PN511
        p[i++] = bin_map(atsc_pn511[j]);

    for (int j = 0; j < 63; j++) // PN63
        p[i++] = bin_map(atsc_pn63[j]);

    for (int j = 0; j < 63; j++) // PN63, toggled on field 2
        p[i++] = bin_map(atsc_pn63[j] ^ mask);

    for (int j = 0; j < 63; j++) // PN63
        p[i++] = bin_map(atsc_pn63[j]);
}

atsc_equalizer_cpu::atsc_equalizer_cpu(const block_args& args) : atsc_equalizer(args)
{
    init_field_sync_common(training_sequence1, 0);
    init_field_sync_common(training_sequence2, 1);

    d_taps.resize(NTAPS, 0.0f);

    d_buff_not_filled = true;

    const int alignment_multiple = volk_get_alignment() / sizeof(float);
    // set_alignment(std::max(1, alignment_multiple));
    set_output_multiple(std::max(1, alignment_multiple));

}

std::vector<float> atsc_equalizer_cpu::taps() const { return d_taps; }

std::vector<float> atsc_equalizer_cpu::data() const
{
    std::vector<float> ret(&data_mem2[0], &data_mem2[ATSC_DATA_SEGMENT_LENGTH - 1]);
    return ret;
}

void atsc_equalizer_cpu::filterN(const float* input_samples,
                             float* output_samples,
                             int nsamples)
{
    for (int j = 0; j < nsamples; j++) {
        output_samples[j] = 0;
        volk_32f_x2_dot_prod_32f(
            &output_samples[j], &input_samples[j], &d_taps[0], NTAPS);
    }
}

void atsc_equalizer_cpu::adaptN(const float* input_samples,
                            const float* training_pattern,
                            float* output_samples,
                            int nsamples)
{
    static const double BETA = 0.00005; // FIXME figure out what this ought to be
                                        // FIXME add gear-shifting

#if 1 // standard lms
    for (int j = 0; j < nsamples; j++) {
        output_samples[j] = 0;
        volk_32f_x2_dot_prod_32f(
            &output_samples[j], &input_samples[j], &d_taps[0], NTAPS);

        float e = output_samples[j] - training_pattern[j];

        // update taps...
        float tmp_taps[NTAPS];
        volk_32f_s32f_multiply_32f(tmp_taps, &input_samples[j], BETA * e, NTAPS);
        volk_32f_x2_subtract_32f(&d_taps[0], &d_taps[0], tmp_taps, NTAPS);
    }

#else // block lms
    int block_size = nsamples; //NTAPS*8;
    int nblocks = nsamples / block_size;
    nsamples = block_size * nblocks;
    float e[block_size];

    for (int j = 0; j < nsamples; j += block_size) {
        
        for (int b = 0; b < block_size; b++) {

            output_samples[j + b] = 0;
            volk_32f_x2_dot_prod_32f(
                &output_samples[j+b], &input_samples[j+b], &d_taps[0], NTAPS);
            e[b] = output_samples[j+b] - training_pattern[j+b];
        }


        float f;
        volk_32f_x2_dot_prod_32f(
                &f, &input_samples[j], &e[0], block_size);

        // update taps...
        float tmp_taps[NTAPS];
        volk_32f_s32f_multiply_32f(tmp_taps, &input_samples[j], BETA * f, NTAPS);
        volk_32f_x2_subtract_32f(&d_taps[0], &d_taps[0], tmp_taps, NTAPS);
    }
#endif

}


work_return_code_t atsc_equalizer_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const float*>(work_input[0].items());
    auto out = static_cast<float*>(work_output[0].items());
    auto plin = static_cast<const plinfo*>(work_input[1].items());
    auto plout = static_cast<plinfo*>(work_output[1].items());

    auto noutput_items = work_output[0].n_items;
    auto ninput_items = work_input[0].n_items;
    if (ninput_items < noutput_items) {
        return work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS;
    }

    int output_produced = 0;
    int i = 0;

    plinfo pli_in;
    if (d_buff_not_filled) {
        memset(&data_mem[0], 0, NPRETAPS * sizeof(float));
        memcpy(&data_mem[NPRETAPS],
               in + i * ATSC_DATA_SEGMENT_LENGTH,
               ATSC_DATA_SEGMENT_LENGTH * sizeof(float));

        d_flags = plin[i].flags();
        d_segno = plin[i].segno();

        d_buff_not_filled = false;
        i++;
    }

    for (; i < noutput_items; i++) {

        memcpy(&data_mem[ATSC_DATA_SEGMENT_LENGTH + NPRETAPS],
               in + i * ATSC_DATA_SEGMENT_LENGTH,
               (NTAPS - NPRETAPS) * sizeof(float));


        if (d_segno == -1) {
            if (d_flags & 0x0010) {
                adaptN(data_mem, training_sequence2, data_mem2, KNOWN_FIELD_SYNC_LENGTH);
            } else if (!(d_flags & 0x0010)) {
                adaptN(data_mem, training_sequence1, data_mem2, KNOWN_FIELD_SYNC_LENGTH);
            }
        } else {
            filterN(data_mem, data_mem2, ATSC_DATA_SEGMENT_LENGTH);

            memcpy(&out[output_produced * ATSC_DATA_SEGMENT_LENGTH],
                   data_mem2,
                   ATSC_DATA_SEGMENT_LENGTH * sizeof(float));

            plout[output_produced++] = plinfo(d_flags, d_segno);
        }

        memcpy(data_mem, &data_mem[ATSC_DATA_SEGMENT_LENGTH], NPRETAPS * sizeof(float));
        memcpy(&data_mem[NPRETAPS],
               in + i * ATSC_DATA_SEGMENT_LENGTH,
               ATSC_DATA_SEGMENT_LENGTH * sizeof(float));

        d_flags = plin[i].flags();
        d_segno = plin[i].segno();
    }

    consume_each(noutput_items, work_input);
    produce_each(output_produced, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace dtv
} // namespace gr