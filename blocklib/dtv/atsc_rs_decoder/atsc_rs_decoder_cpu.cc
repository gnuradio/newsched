#include "atsc_rs_decoder_cpu.hh"
#include "atsc_types.hh"
#include <gnuradio/dtv/atsc_consts.hh>

namespace gr {
namespace dtv {

static const int rs_init_symsize = 8;
static const int rs_init_gfpoly = 0x11d;
static const int rs_init_fcr = 0;  // first consecutive root
static const int rs_init_prim = 1; // primitive is 1 (alpha)
static const int rs_init_nroots = 20;

static const int N = (1 << rs_init_symsize) - 1; // 255

static const int amount_of_pad = N - ATSC_MPEG_RS_ENCODED_LENGTH; // 48

atsc_rs_decoder::sptr atsc_rs_decoder::make_cpu(const block_args& args) { return std::make_shared<atsc_rs_decoder_cpu>(args); }

atsc_rs_decoder_cpu::atsc_rs_decoder_cpu(const block_args& args) : atsc_rs_decoder(args)
{
    d_rs = init_rs_char(
        rs_init_symsize, rs_init_gfpoly, rs_init_fcr, rs_init_prim, rs_init_nroots);
    assert(d_rs != 0);
    d_nerrors_corrected_count = 0;
    d_bad_packet_count = 0;
    d_total_packets = 0;
}

int atsc_rs_decoder_cpu::decode(uint8_t* out, const uint8_t* in)
{
    unsigned char tmp[N];
    int ncorrections;

    // assert((int)(amount_of_pad + sizeof(in.data)) == N);
    assert((int)(amount_of_pad + ATSC_MPEG_RS_ENCODED_LENGTH) == N);

    // add missing prefix zero padding to message
    memset(tmp, 0, amount_of_pad);
    memcpy(&tmp[amount_of_pad], in, ATSC_MPEG_RS_ENCODED_LENGTH);

    // correct message...
    ncorrections = decode_rs_char(d_rs, tmp, 0, 0);

    // copy corrected message to output, skipping prefix zero padding
    memcpy(out, &tmp[amount_of_pad], ATSC_MPEG_PKT_LENGTH);

    return ncorrections;
}

atsc_rs_decoder_cpu::~atsc_rs_decoder_cpu()
{
    if (d_rs)
        free_rs_char(d_rs);
    d_rs = 0;
}

int atsc_rs_decoder_cpu::num_errors_corrected() const { return d_nerrors_corrected_count; }

int atsc_rs_decoder_cpu::num_bad_packets() const { return d_bad_packet_count; }

int atsc_rs_decoder_cpu::num_packets() const { return d_total_packets; }


work_return_code_t atsc_rs_decoder_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const uint8_t*>(work_input[0].items());
    auto out = static_cast<uint8_t*>(work_output[0].items());
    auto plin = static_cast<const plinfo*>(work_input[1].items());
    auto plout = static_cast<plinfo*>(work_output[1].items());
    auto noutput_items = work_output[0].n_items;

    std::vector<tag_t> tags;

    for (int i = 0; i < noutput_items; i++) {
        assert(plin[i].regular_seg_p());

        plout[i] = plin[i];

        int nerrors_corrected =
            decode(&out[i * ATSC_MPEG_PKT_LENGTH], &in[i * ATSC_MPEG_RS_ENCODED_LENGTH]);
        plout[i].set_transport_error(nerrors_corrected == -1);
        if (nerrors_corrected == -1) {
            d_bad_packet_count++;
            d_nerrors_corrected_count += 10; // lower bound estimate; most this RS can fix
        } else {
            d_nerrors_corrected_count += nerrors_corrected;
        }

        d_total_packets++;
    }

    produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}

} // namespace dtv
} // namespace gr