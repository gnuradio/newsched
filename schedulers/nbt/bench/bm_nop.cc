#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocks/nop.hh>
#include <gnuradio/blocks/nop_head.hh>
#include <gnuradio/blocks/null_sink.hh>
#include <gnuradio/blocks/nop_source.hh>
#include <gnuradio/flowgraph.hh>
#include <gnuradio/realtime.hh>
#include <gnuradio/schedulers/nbt/scheduler_nbt.hh>
#include <gnuradio/buffer_cpu_simple.hh>
#include <gnuradio/buffer_cpu_vmcirc.hh>

#include <iostream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace gr;

int main(int argc, char* argv[])
{
    uint64_t samples;
    int nblocks;
    int nthreads;
    int veclen;
    int buffer_type;
    bool rt_prio = false;

    po::options_description desc("Basic Test Flow Graph");
    desc.add_options()("help,h", "display help")(
        "samples",
        po::value<uint64_t>(&samples)->default_value(15000000),
        "Number of samples")(
        "veclen", po::value<int>(&veclen)->default_value(1), "Vector Length")(
        "nblocks", po::value<int>(&nblocks)->default_value(1), "Number of copy blocks")(
        "nthreads",
        po::value<int>(&nthreads)->default_value(0),
        "Number of threads (0: tpb")(
        "buffer",
        po::value<int>(&buffer_type)->default_value(1),
        "Buffer Type (0:simple, 1:vmcirc, 2:cuda, 3:cuda_pinned")(
        "rt_prio", "Enable Real-time priority");

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


    if (rt_prio && gr::enable_realtime_scheduling() != RT_OK) {
        std::cout << "Error: failed to enable real-time scheduling." << std::endl;
    }

    {
        auto src = blocks::nop_source::make({sizeof(gr_complex) * veclen});
        auto head = blocks::nop_head::make({sizeof(gr_complex) * veclen, samples / veclen});
        auto snk = blocks::null_sink::make({sizeof(gr_complex) * veclen});
        std::vector<blocks::nop::sptr> blks(nblocks);
        for (int i = 0; i < nblocks; i++) {
            blks[i] = blocks::nop::make({sizeof(gr_complex) * veclen});
        }
        flowgraph_sptr fg(new flowgraph());

        if (buffer_type == 0) {
            fg->connect(src, 0, head, 0);
            fg->connect(head, 0, blks[0], 0);
            for (int i = 0; i < nblocks - 1; i++) {
                fg->connect(blks[i], 0, blks[i + 1], 0);
            }
            fg->connect(blks[nblocks - 1], 0, snk, 0);

        } else {
            fg->connect(src, 0, head, 0)->set_custom_buffer(BUFFER_CPU_VMCIRC_ARGS);
            fg->connect(head, 0, blks[0], 0)->set_custom_buffer(BUFFER_CPU_VMCIRC_ARGS);
            for (int i = 0; i < nblocks - 1; i++) {
                fg->connect(blks[i], 0, blks[i + 1], 0)->set_custom_buffer(BUFFER_CPU_VMCIRC_ARGS);
            }
            fg->connect(blks[nblocks - 1], 0, snk, 0)->set_custom_buffer(BUFFER_CPU_VMCIRC_ARGS);
        }

        auto sched = schedulers::scheduler_nbt::make("nbt", 32768);
        fg->add_scheduler(sched);

        if (buffer_type == 1) {
            sched->set_default_buffer_factory(BUFFER_CPU_VMCIRC_ARGS);
        }

        if (nthreads > 0) {
            int blks_per_thread = nblocks / nthreads;

            for (int i = 0; i < nthreads; i++) {
                std::vector<block_sptr> block_group;
                if (i == 0) {
                    block_group.push_back(src);
                    block_group.push_back(head);
                }

                for (int j = 0; j < blks_per_thread; j++) {
                    block_group.push_back(blks[i * blks_per_thread + j]);
                }

                if (i == nthreads - 1) {
                    for (int j = 0; j < (nblocks - nthreads * blks_per_thread); j++) {
                        block_group.push_back(blks[(i + 1) * blks_per_thread + j]);
                    }
                    block_group.push_back(snk);
                }
                sched->add_block_group(block_group);
            }
        }

        fg->validate();

        auto t1 = std::chrono::steady_clock::now();

        fg->start();
        fg->wait();

        auto t2 = std::chrono::steady_clock::now();
        auto time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9;

        std::cout << "[PROFILE_TIME]" << time << "[PROFILE_TIME]" << std::endl;
    }
}
