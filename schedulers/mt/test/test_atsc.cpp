
#include <gnuradio/fileio/file_source.hpp>
#include <gnuradio/fileio/file_sink.hpp>
#include <gnuradio/streamops/interleaved_short_to_complex.hpp>
#include <gnuradio/filter/dc_blocker.hpp>
#include <gnuradio/analog/agc_blk.hpp>
#include <gnuradio/dtv/atsc_fpll.hpp>
#include <gnuradio/dtv/atsc_sync.hpp>
#include <gnuradio/dtv/atsc_fs_checker.hpp>
#include <gnuradio/dtv/atsc_equalizer.hpp>
#include <gnuradio/dtv/atsc_viterbi_decoder.hpp>

#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/mt/scheduler_mt.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    double sps = 1.1;
    double atsc_sym_rate = 4.5e6/286*684;
    double oversampled_rate = atsc_sym_rate * sps;

    auto src = fileio::file_source::make(2*sizeof(uint16_t), argv[1], false);
    auto is2c = streamops::interleaved_short_to_complex::make(false, 32768.0);
    auto fpll = dtv::atsc_fpll::make(oversampled_rate);
    auto dcb = filter::dc_blocker<float>::make(4096, true);
    auto agc = analog::agc_blk<float>::make(1e-5, 4.0, 1.0);
    auto sync = dtv::atsc_sync::make(oversampled_rate);

    char filename_out[1024];
    auto fn = tmpnam(filename_out);
    std::cout << fn << std::endl;
    // auto snk = fileio::file_sink::make(sizeof(gr_complex), fn);
    auto snk = fileio::file_sink::make(sizeof(float), fn);

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, is2c, 0);
    fg->connect(is2c, 0, fpll, 0);
    fg->connect(fpll, 0, dcb, 0);
    fg->connect(dcb, 0, agc, 0);
    fg->connect(agc, 0, sync, 0);
    fg->connect(sync, 0, snk, 0);

    std::shared_ptr<schedulers::scheduler_mt> sched(new schedulers::scheduler_mt());
    fg->set_scheduler(sched);
    fg->validate();

    fg->start();
    fg->wait();
}