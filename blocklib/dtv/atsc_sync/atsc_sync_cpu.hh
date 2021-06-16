#pragma once

#include <gnuradio/dtv/atsc_sync.hh>
#include <gnuradio/dtv/atsc_consts.hh>
#include <gnuradio/filter/mmse_fir_interpolator_ff.hh>
#include <gnuradio/filter/single_pole_iir.hh>

namespace gr {
namespace dtv {

class atsc_sync_cpu : public atsc_sync
{
public:
    atsc_sync_cpu(const block_args& args);
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;
    void reset();

private:
    gr::filter::kernel::single_pole_iir<float, float, float> d_loop; // ``VCO'' loop filter
    gr::filter::kernel::mmse_fir_interpolator_ff d_interp;

    double d_rx_clock_to_symbol_freq;
    int d_si;
    double d_w;  // ratio of PERIOD of Tx to Rx clocks
    double d_mu; // fractional delay [0,1]
    int d_incr;

    float d_sample_mem[ATSC_DATA_SEGMENT_LENGTH];
    float d_data_mem[ATSC_DATA_SEGMENT_LENGTH];

    double d_timing_adjust;
    int d_counter; // free running mod 832 counter
    int d_symbol_index;
    bool d_seg_locked;
    unsigned char d_sr; // 4 bit shift register
    signed char d_integrator[ATSC_DATA_SEGMENT_LENGTH];
    int d_output_produced;
};

} // namespace dtv
} // namespace gr