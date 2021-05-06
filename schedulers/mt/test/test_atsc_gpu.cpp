
#include <gnuradio/analog/agc_blk.hpp>
#include <gnuradio/dtv/atsc_deinterleaver.hpp>
#include <gnuradio/dtv/atsc_derandomizer.hpp>
#include <gnuradio/dtv/atsc_equalizer.hpp>
#include <gnuradio/dtv/atsc_fpll.hpp>
#include <gnuradio/dtv/atsc_fs_checker.hpp>
#include <gnuradio/dtv/atsc_rs_decoder.hpp>
#include <gnuradio/dtv/atsc_sync.hpp>
#include <gnuradio/dtv/atsc_viterbi_decoder.hpp>
#include <gnuradio/dtv/cuda/atsc_equalizer_cuda.hpp>
#include <gnuradio/dtv/cuda/atsc_fs_checker_cuda.hpp>
#include <gnuradio/dtv/cuda/atsc_sync_cuda.hpp>
#include <gnuradio/dtv/cuda/atsc_viterbi_decoder_cuda.hpp>
#include <gnuradio/fileio/file_sink.hpp>
#include <gnuradio/fileio/file_source.hpp>
#include <gnuradio/filter/dc_blocker.hpp>
#include <gnuradio/streamops/interleaved_short_to_complex.hpp>

#include <gnuradio/blocks/null_sink.hpp>

#include <gnuradio/cudabuffer.hpp>
#include <gnuradio/cudabuffer_pinned.hpp>
#include <gnuradio/dtv/atsc_plinfo.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/mt/scheduler_mt.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    double sps = 1.1;
    double atsc_sym_rate = 4.5e6 / 286 * 684;
    double oversampled_rate = atsc_sym_rate * sps;


    flowgraph_sptr fg(new flowgraph());
    auto sched = schedulers::scheduler_mt::make("sched", 32768 * 16);

#if 0
    auto src = fileio::file_source::make(2 * sizeof(uint16_t), argv[1], false);
    // auto src = fileio::file_source::make(sizeof(float)*1, argv[1], false);
    // auto src = fileio::file_source::make(sizeof(float)*832, argv[1], false);
    auto is2c = streamops::interleaved_short_to_complex::make(false, 32768.0);
    auto fpll = dtv::atsc_fpll::make(oversampled_rate);
    auto dcb = filter::dc_blocker<float>::make(4096, true);
    auto agc = analog::agc_blk<float>::make(1e-5, 4.0, 1.0);
    auto sync = dtv::atsc_sync_cuda::make(oversampled_rate);
    auto fschk = dtv::atsc_fs_checker_cuda::make();
    // auto fschk = dtv::atsc_fs_checker::make();
    auto eq = dtv::atsc_equalizer_cuda::make();
    // auto eq = dtv::atsc_equalizer::make();
    auto vit = dtv::atsc_viterbi_decoder_cuda::make();
    auto dei = dtv::atsc_deinterleaver::make();
    auto rsd = dtv::atsc_rs_decoder::make();
    auto der = dtv::atsc_derandomizer::make();

    auto snk = fileio::file_sink::make(sizeof(uint8_t) * 188, "/tmp/mpeg.live.ts");
    // auto null = blocks::null_sink::make(4); // plinfo

    fg->connect(src, 0, is2c, 0);
    fg->connect(is2c, 0, fpll, 0);
    fg->connect(fpll, 0, dcb, 0)->set_custom_buffer(VMCIRC_BUFFER_ARGS);
    fg->connect(dcb, 0, agc, 0);
    fg->connect(agc, 0, sync, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D->set_max_buffer_read(32*832));
    // fg->connect(agc, 0, sync, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS->set_max_buffer_read(32*832));


    // fg->connect(src, 0, sync, 0);
    // fg->connect(sync, 0, snk, 0);
    fg->connect(sync, 0, fschk, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D); //->set_max_buffer_fill(32));
    // fg->connect(sync, 0, fschk, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);

    fg->connect(fschk, 0, eq, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D); //->set_max_buffer_fill(32));
    // fg->connect(eq, 0, vit, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D);
    // fg->connect(fschk, 0, eq, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D);
    fg->connect(eq, 0, vit, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D); 
    // fg->connect(fschk, 0, eq, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    // fg->connect(eq, 0, vit, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);

    fg->connect(fschk, 1, eq, 1);


    // fg->connect(eq,0,snkeq,0);
    fg->connect(eq, 1, vit, 1);

    fg->connect(vit, 0, dei, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2H->set_max_buffer_fill(12));
    // fg->connect(vit, 0, dei, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    fg->connect(vit, 1, dei, 1);

    fg->connect(dei, 0, rsd, 0);
    fg->connect(dei, 1, rsd, 1);
    fg->connect(rsd, 0, der, 0);
    fg->connect(rsd, 1, der, 1);
    fg->connect(der, 0, snk, 0);
#elif 0
    // auto src = fileio::file_source::make(sizeof(float), "/tmp/fpll_out.f32", false);
    auto src = fileio::file_source::make(sizeof(float), "/tmp/fpll_out.f32", false);
    auto dcb = filter::dc_blocker<float>::make(4096, true);
    auto agc = analog::agc_blk<float>::make(1e-5, 4.0, 1.0);
    auto sync = dtv::atsc_sync_cuda::make(oversampled_rate);
    // auto fschk = dtv::atsc_fs_checker_cuda::make();
    auto fschk = dtv::atsc_fs_checker::make();
    auto eq = dtv::atsc_equalizer_cuda::make();
    // auto eq = dtv::atsc_equalizer::make();
    auto vit = dtv::atsc_viterbi_decoder_cuda::make();
    auto dei = dtv::atsc_deinterleaver::make();
    auto rsd = dtv::atsc_rs_decoder::make();
    auto der = dtv::atsc_derandomizer::make();

    auto snk = fileio::file_sink::make(sizeof(uint8_t) * 188, "/tmp/mpeg.live.ts");
    // auto null = blocks::null_sink::make(4); // plinfo

    fg->connect(src, 0, dcb, 0);
    fg->connect(dcb, 0, agc, 0);

    fg->connect(agc, 0, sync, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS); //->set_max_buffer_read(32*832));
    fg->connect(sync, 0, fschk, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    fg->connect(fschk, 0, eq, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    fg->connect(eq, 0, vit, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    fg->connect(vit, 0, dei, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS); //->set_max_buffer_fill(12));

    // fg->connect(agc, 0, sync, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D); //->set_max_buffer_read(32*832));
    // fg->connect(sync, 0, fschk, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D); //->set_max_buffer_fill(32));
    // fg->connect(fschk, 0, eq, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D); //->set_max_buffer_fill(32));
    // fg->connect(eq, 0, vit, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D); 
    // fg->connect(vit, 0, dei, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2H); //->set_max_buffer_fill(12));

    fg->connect(fschk, 1, eq, 1);


    // fg->connect(eq,0,snkeq,0);
    fg->connect(eq, 1, vit, 1);

    
    // fg->connect(vit, 0, dei, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    fg->connect(vit, 1, dei, 1);

    fg->connect(dei, 0, rsd, 0);
    fg->connect(dei, 1, rsd, 1);
    fg->connect(rsd, 0, der, 0);
    fg->connect(rsd, 1, der, 1);
    fg->connect(der, 0, snk, 0);
#elif 1
    auto src = fileio::file_source::make(832*sizeof(float), "/tmp/sync_out.f32", false);
    // auto dcb = filter::dc_blocker<float>::make(4096, true);
    // auto agc = analog::agc_blk<float>::make(1e-5, 4.0, 1.0);
    // auto sync = dtv::atsc_sync_cuda::make(oversampled_rate);
    auto fschk = dtv::atsc_fs_checker_cuda::make();
    // auto fschk = dtv::atsc_fs_checker::make();
    auto eq = dtv::atsc_equalizer_cuda::make();
    // auto eq = dtv::atsc_equalizer::make();
    auto vit = dtv::atsc_viterbi_decoder_cuda::make();
    auto dei = dtv::atsc_deinterleaver::make();
    auto rsd = dtv::atsc_rs_decoder::make();
    auto der = dtv::atsc_derandomizer::make();

    auto snk = fileio::file_sink::make(sizeof(uint8_t) * 188, "/tmp/mpeg.live.ts");
    // auto snk_fs = fileio::file_sink::make(832*sizeof(float), "/tmp/ns_fs_out.f32");
    // auto null = blocks::null_sink::make(4); // plinfo

    // fg->connect(src, 0, dcb, 0);
    // fg->connect(dcb, 0, agc, 0);

    // fg->connect(agc, 0, sync, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS); //->set_max_buffer_read(32*832));
    fg->connect(src, 0, fschk, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    fg->connect(fschk, 0, eq, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    fg->connect(eq, 0, vit, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    fg->connect(vit, 0, dei, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS); //->set_max_buffer_fill(12));

    // fg->connect(agc, 0, sync, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D); //->set_max_buffer_read(32*832));
    // fg->connect(src, 0, fschk, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D); //->set_max_buffer_fill(32));
    // fg->connect(fschk, 0, eq, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D); //->set_max_buffer_fill(32));
    // fg->connect(eq, 0, vit, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D); 
    // fg->connect(vit, 0, dei, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2H); //->set_max_buffer_fill(12));

    fg->connect(fschk, 1, eq, 1);

    // fg->connect(fschk, 0, snk_fs, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS); //->set_max_buffer_fill(32));
    // fg->connect(eq,0,snkeq,0);
    fg->connect(eq, 1, vit, 1);

    
    // fg->connect(vit, 0, dei, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    fg->connect(vit, 1, dei, 1);

    fg->connect(dei, 0, rsd, 0);
    fg->connect(dei, 1, rsd, 1);
    fg->connect(rsd, 0, der, 0);
    fg->connect(rsd, 1, der, 1);
    fg->connect(der, 0, snk, 0);
#else
    // auto src = fileio::file_source::make(sizeof(float), "/tmp/ns_fpll_out.dat", false);
    auto src = fileio::file_source::make(sizeof(float), "/tmp/short_fpll.dat", false);
    auto dcb = filter::dc_blocker<float>::make(4096, true);
    auto agc = analog::agc_blk<float>::make(1e-5, 4.0, 1.0);
    auto sync = dtv::atsc_sync_cuda::make(oversampled_rate);
    auto fschk = dtv::atsc_fs_checker_cuda::make();
    auto eq = dtv::atsc_equalizer_cuda::make();
    // auto eq = dtv::atsc_equalizer::make();
    auto vit = dtv::atsc_viterbi_decoder_cuda::make();
    auto dei = dtv::atsc_deinterleaver::make();
    auto rsd = dtv::atsc_rs_decoder::make();
    auto der = dtv::atsc_derandomizer::make();

    auto snk = fileio::file_sink::make(sizeof(uint8_t) * 188, "/tmp/mpeg.live.ts");
    // auto null = blocks::null_sink::make(4); // plinfo

    fg->connect(src, 0, dcb, 0);
    fg->connect(dcb, 0, agc, 0);
    fg->connect(agc, 0, sync, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D);


    // fg->connect(src, 0, sync, 0);
    // fg->connect(sync, 0, snk, 0);
    fg->connect(sync, 0, fschk, 0)
        ->set_custom_buffer(
            CUDA_BUFFER_ARGS_D2D); //->set_custom_buffer(CUDA_BUFFER_ARGS_D2D);

    // fg->connect(fschk, 0, eq, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2H);
    // fg->connect(eq, 0, vit, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D);
    fg->connect(fschk, 0, eq, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D);
    fg->connect(eq, 0, vit, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D);

    fg->connect(fschk, 1, eq, 1);


    // fg->connect(eq,0,snkeq,0);
    fg->connect(eq, 1, vit, 1);

    fg->connect(vit, 0, dei, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2H);
    fg->connect(vit, 1, dei, 1);

    fg->connect(dei, 0, rsd, 0);
    fg->connect(dei, 1, rsd, 1);
    fg->connect(rsd, 0, der, 0);
    fg->connect(rsd, 1, der, 1);
    fg->connect(der, 0, snk, 0);
#endif


    // sched->add_block_group({ dei, rsd, der, snk });
    // sched->add_block_group({src,is2c});
    // sched->add_block_group({src, dcb, agc });
    // sched->add_block_group({sync, fschk, eq, vit});
    // sched->add_block_group(
    // { src, is2c, fpll, dcb, agc, sync, fschk, eq, vit, dei, rsd, der, snk });

    // auto dbg_snk1 = fileio::file_sink::make(sizeof(float), "/tmp/ns_fpll_out.dat",
    // false); auto dbg_snk2 = fileio::file_sink::make(832*sizeof(float),
    // "/tmp/ns_sync_out.dat", false); auto dbg_snk3 =
    // fileio::file_sink::make(832*sizeof(float), "/tmp/ns_fs_out.dat", false); auto
    // dbg_snk4 = fileio::file_sink::make(832*sizeof(float), "/tmp/ns_eq_out.dat", false);
    // auto dbg_snk5 = fileio::file_sink::make(207*sizeof(uint8_t), "/tmp/ns_vit_out.dat",
    // false); auto dbg_snk6 = fileio::file_sink::make(sizeof(dtv::plinfo),
    // "/tmp/fs_plout.dat", false); auto dbg_snk7 =
    // fileio::file_sink::make(sizeof(dtv::plinfo), "/tmp/eq_plout.dat", false);

    // fg->connect(fpll, 0, dbg_snk1, 0);
    // fg->connect(sync, 0, dbg_snk2, 0);
    // fg->connect(fschk, 0, dbg_snk3, 0);
    // fg->connect(fschk, 1, dbg_snk6, 0);
    // fg->connect(eq, 0, dbg_snk4, 0);
    // fg->connect(eq, 1, dbg_snk7, 0);
    // fg->connect(vit, 0, dbg_snk5, 0);

    fg->set_scheduler(sched);
    fg->validate();

    fg->start();
    fg->wait();
}