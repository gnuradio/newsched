#include "atsc_sync_cpu.hh"

namespace gr {
namespace dtv {


static const double LOOP_FILTER_TAP = 0.0005; // 0.0005 works
static const double ADJUSTMENT_GAIN = 1.0e-5 / (10 * ATSC_DATA_SEGMENT_LENGTH);
static const int SYMBOL_INDEX_OFFSET = 3;
static const int MIN_SEG_LOCK_CORRELATION_VALUE = 5;
static const signed char SSI_MIN = -16;
static const signed char SSI_MAX = 15;


atsc_sync::sptr atsc_sync::make_cpu(const block_args& args) { return std::make_shared<atsc_sync_cpu>(args); }

atsc_sync_cpu::atsc_sync_cpu(const block_args& args) : atsc_sync(args),
      d_rx_clock_to_symbol_freq(args.rate / ATSC_SYMBOL_RATE),
      d_si(0)
{
    d_loop.set_taps(LOOP_FILTER_TAP);
    reset();
}

void atsc_sync_cpu::reset()
{
    d_w = d_rx_clock_to_symbol_freq;
    d_mu = 0.5;

    d_timing_adjust = 0;
    d_counter = 0;
    d_symbol_index = 0;
    d_seg_locked = false;

    d_sr = 0;

    memset(d_sample_mem,
           0,
           ATSC_DATA_SEGMENT_LENGTH * sizeof(*d_sample_mem)); // (float)0 = 0x00000000
    memset(d_data_mem,
           0,
           ATSC_DATA_SEGMENT_LENGTH * sizeof(*d_data_mem)); // (float)0 = 0x00000000
    memset(d_integrator,
           SSI_MIN,
           ATSC_DATA_SEGMENT_LENGTH * sizeof(*d_integrator)); // signed char
}


work_return_code_t atsc_sync_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const float*>(work_input[0].items());
    auto out = static_cast<float*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;
    auto ninput_items = work_input[0].n_items;

    // amount actually consumed
    d_si = 0;

    // Because this is a general block, we must do some forecasting
    auto min_items = static_cast<int>(noutput_items * d_rx_clock_to_symbol_freq *
                                      ATSC_DATA_SEGMENT_LENGTH) +
                     1500 - 1;
    if (work_input[0].n_items < min_items) {
        consume_each(0,work_input);
        return work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS;
    }


    float interp_sample;


    for (d_output_produced = 0; d_output_produced < noutput_items &&
                                (d_si + (int)d_interp.ntaps()) < ninput_items;) {
        // First we interpolate a sample from input to work with
        interp_sample = d_interp.interpolate(&in[d_si], d_mu);

        // Apply our timing adjustment slowly over several samples
        d_mu += ADJUSTMENT_GAIN * 1e3 * d_timing_adjust;

        double s = d_mu + d_w;
        double float_incr = floor(s);
        d_mu = s - float_incr;
        d_incr = (int)float_incr;

        assert(d_incr >= 1 && d_incr <= 3);
        d_si += d_incr;

        // Remember the sample at this count position
        d_sample_mem[d_counter] = interp_sample;

        // Is the sample positive or negative?
        int bit = (interp_sample < 0 ? 0 : 1);

        // Put the sign bit into our shift register
        d_sr = ((bit & 1) << 3) | (d_sr >> 1);

        // When +,-,-,+ (0x9, 1001) samples show up we have likely found a segment
        // sync, it is more likely the segment sync will show up at about the same
        // spot every ATSC_DATA_SEGMENT_LENGTH samples so we add some weight
        // to this spot every pass to prevent random +,-,-,+ symbols from
        // confusing our synchronizer
        d_integrator[d_counter] += ((d_sr == 0x9) ? +2 : -1);
        if (d_integrator[d_counter] < SSI_MIN)
            d_integrator[d_counter] = SSI_MIN;
        if (d_integrator[d_counter] > SSI_MAX)
            d_integrator[d_counter] = SSI_MAX;

        d_symbol_index++;
        if (d_symbol_index >= ATSC_DATA_SEGMENT_LENGTH)
            d_symbol_index = 0;

        d_counter++;
        if (d_counter >= ATSC_DATA_SEGMENT_LENGTH) { // counter just wrapped...
            int best_correlation_value = d_integrator[0];
            int best_correlation_index = 0;

            for (int i = 1; i < ATSC_DATA_SEGMENT_LENGTH; i++)
                if (d_integrator[i] > best_correlation_value) {
                    best_correlation_value = d_integrator[i];
                    best_correlation_index = i;
                }

            d_seg_locked = best_correlation_value >= MIN_SEG_LOCK_CORRELATION_VALUE;

            // the coefficients are -1,-1,+1,+1
            // d_timing_adjust = d_sample_mem[best_correlation_index - 3] +
            //                   d_sample_mem[best_correlation_index - 2] -
            //                   d_sample_mem[best_correlation_index - 1] -
            //                   d_sample_mem[best_correlation_index];

            int corr_count = best_correlation_index;

            d_timing_adjust = -d_sample_mem[corr_count--];
            if (corr_count < 0)
                corr_count = ATSC_DATA_SEGMENT_LENGTH - 1;
            d_timing_adjust -= d_sample_mem[corr_count--];
            if (corr_count < 0)
                corr_count = ATSC_DATA_SEGMENT_LENGTH - 1;
            d_timing_adjust += d_sample_mem[corr_count--];
            if (corr_count < 0)
                corr_count = ATSC_DATA_SEGMENT_LENGTH - 1;
            d_timing_adjust += d_sample_mem[corr_count--];

            d_symbol_index = SYMBOL_INDEX_OFFSET - 1 - best_correlation_index;
            if (d_symbol_index < 0)
                d_symbol_index += ATSC_DATA_SEGMENT_LENGTH;

            d_counter = 0;
        }

        // If we are locked we can start filling and producing data packets
        // Due to the way we lock the first data packet will almost always be
        // half full, this is OK because the fs_checker will not let packets though
        // until a non-corrupted field packet is found
        if (d_seg_locked) {
            d_data_mem[d_symbol_index] = interp_sample;

            if (d_symbol_index >= (ATSC_DATA_SEGMENT_LENGTH - 1)) {
                memcpy(&out[d_output_produced * ATSC_DATA_SEGMENT_LENGTH],
                       d_data_mem,
                       ATSC_DATA_SEGMENT_LENGTH * sizeof(float));
                d_output_produced++;
            }
        }
    }

    work_input[0].n_consumed = d_si;
    work_output[0].n_produced = d_output_produced;
    return work_return_code_t::WORK_OK;
}


} // namespace dtv
} // namespace gr