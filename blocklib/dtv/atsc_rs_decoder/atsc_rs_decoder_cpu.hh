#pragma once

#include <gnuradio/dtv/atsc_rs_decoder.hh>

extern "C" {
#include <gnuradio/fec/rs.h>
}


namespace gr {
namespace dtv {

class atsc_rs_decoder_cpu : public atsc_rs_decoder
{
public:
    atsc_rs_decoder_cpu(const block_args& args);
    ~atsc_rs_decoder_cpu() override;
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

    /*!
     * Returns the number of errors corrected by the decoder.
     */
    int num_errors_corrected() const;

    /*!
     * Returns the number of bad packets rejected by the decoder.
     */
    int num_bad_packets() const;

    /*!
     * Returns the total number of packets seen by the decoder.
     */
    int num_packets() const;

    /*!
     * Decode RS encoded packet.
     * \returns a count of corrected symbols, or -1 if the block was uncorrectible.
     */
    int decode(uint8_t* out, const uint8_t* in);

private:
    int d_nerrors_corrected_count;
    int d_bad_packet_count;
    int d_total_packets;
    void* d_rs;
};

} // namespace dtv
} // namespace gr