#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/head.hpp>
#include <gnuradio/blocklib/blocks/null_sink.hpp>
#include <gnuradio/blocklib/blocks/null_source.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/blocklib/cuda/fft.hpp>
#include <gnuradio/domain_adapter_direct.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/logging.hpp>
#include <gnuradio/realtime.hpp>
#include <gnuradio/schedulers/mt/scheduler_mt.hpp>

#include <gnuradio/cudabuffer.hpp>
#include <gnuradio/cudabuffer_pinned.hpp>
#include <gnuradio/simplebuffer.hpp>
#include <gnuradio/vmcirc_buffer.hpp>
#include <iostream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace gr;

int main(int argc, char* argv[])
{
    int run;
    uint64_t samples;
    int nsched;
    int mem_model;
    int batch_size;
    int nblocks;
    int fftsize;
    bool rt_prio = false;
    bool machine_readable = false;

    po::options_description desc("CUDA FFT Benchmarking Flowgraph");
    desc.add_options()("help,h", "display help")("samples,N",
        po::value<uint64_t>(&samples)->default_value(15000000),
        "Number of samples")(
        "nblocks,b", po::value<int>(&nblocks)->default_value(4), "Num FFT Blocks")(
        "fftsize,b", po::value<int>(&fftsize)->default_value(1024), "FFT Size")(
        "memmodel,m", po::value<int>(&mem_model)->default_value(0), "Memory Model")(
        "batchsize,s", po::value<int>(&batch_size)->default_value(1), "Batch Size");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    if (vm.count("rt_prio")) {
        rt_prio = true;
    }


    std::vector<cuda::fft::sptr> blks(nblocks);
    for (int i = 0; i < nblocks; i++) {
        blks[i] = cuda::fft::make(fftsize, true, false, batch_size);
    }

    std::vector<gr_complex> input_data(samples);
    for (auto i = 0; i < samples; i++)
        input_data[i] = gr_complex(i % 256, 256 - i % 256);

    auto src = blocks::null_source::make(sizeof(gr_complex) * batch_size);
    auto snk = blocks::null_sink::make(sizeof(gr_complex) * batch_size);
    auto head = blocks::head::make(sizeof(gr_complex) * fftsize, samples / fftsize);

    auto fg = flowgraph::make();

    fg->connect(src, 0, head, 0, VMCIRC_BUFFER_ARGS);
    auto sched = schedulers::scheduler_mt::make(
        "sched",
        32768); // This sizing should be handled in buffer_managment but it is not yet
    sched->set_default_buffer_factory(VMCIRC_BUFFER_ARGS);
    fg->set_scheduler(sched);
    if (mem_model == 0) {
        fg->connect(head, 0, blks[0], 0, CUDA_BUFFER_ARGS_H2D);
        for (int i = 0; i < nblocks - 1; i++) {
            fg->connect(blks[i], 0, blks[i + 1], 0, CUDA_BUFFER_ARGS_D2D);
        }
        fg->connect(blks[nblocks - 1], 0, snk, 0, CUDA_BUFFER_ARGS_D2H);

    } else {
        fg->connect(head, 0, blks[0], 0, CUDA_BUFFER_PINNED_ARGS);
        for (int i = 0; i < nblocks - 1; i++) {
            fg->connect(blks[i], 0, blks[i + 1], 0, CUDA_BUFFER_PINNED_ARGS);
        }
        fg->connect(blks[nblocks - 1], 0, snk, 0, CUDA_BUFFER_PINNED_ARGS);
    }

    fg->validate();

    if (rt_prio && gr::enable_realtime_scheduling() != gr::rt_status_t::RT_OK)
        std::cout << "Unable to enable realtime scheduling " << std::endl;

    auto t1 = std::chrono::steady_clock::now();
    fg->start();
    fg->wait();

    auto t2 = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9;

    std::cout << "[PROFILE_TIME]" << time << "[PROFILE_TIME]" << std::endl;
}
