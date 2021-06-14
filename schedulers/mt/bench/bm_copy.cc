#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocks/copy.hh>
#include <gnuradio/blocks/head.hh>
#include <gnuradio/blocks/null_sink.hh>
#include <gnuradio/blocks/null_source.hh>
#include <gnuradio/flowgraph.hh>
#include <gnuradio/realtime.hh>
#include <gnuradio/schedulers/mt/scheduler_mt.hh>
#include <gnuradio/buffer_cpu_simple.hh>
#include <gnuradio/buffer_cpu_vmcirc.hh>

#include <iostream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace gr;

int main(int argc, char* argv[])
{
    uint64_t samples;
    unsigned int nblocks;
    unsigned int nthreads;
    int veclen;
    int buffer_type;
    int buffer_size;
    bool rt_prio = false;
   
    std::vector<unsigned int> cpu_affinity;

    po::options_description desc("Basic Test Flow Graph");
    desc.add_options()("help,h", "display help")(
        "samples",
        po::value<uint64_t>(&samples)->default_value(15000000),
        "Number of samples")(
        "veclen", po::value<int>(&veclen)->default_value(1), "Vector Length")(
        "nblocks", po::value<unsigned int>(&nblocks)->default_value(1), "Number of copy blocks")(
        "nthreads",
        po::value<unsigned int>(&nthreads)->default_value(0),
        "Number of threads (0: tpb")(
        "buffer",
        po::value<int>(&buffer_type)->default_value(1),
        "Buffer Type (0:simple, 1:vmcirc, 2:cuda, 3:cuda_pinned")(
        "buffer_size",
        po::value<int>(&buffer_size)->default_value(32768),
        "Buffer Size in bytes")(
        "rt_prio", "Enable Real-time priority")(
        "cpus", po::value<std::vector<unsigned int>>()->multitoken(), "Pin threads to CPUs (if nthreads > 0, will pin to 0,1,..,N"
        );

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

    if (vm.count("cpus")) {
      cpu_affinity = vm["cpus"].as<std::vector<unsigned int>>();
    }

    if (rt_prio && gr::enable_realtime_scheduling() != RT_OK) {
        std::cout << "Error: failed to enable real-time scheduling." << std::endl;
    }

    {
        auto src = blocks::null_source::make({sizeof(gr_complex) * veclen});
        auto head = blocks::head::make_cpu({sizeof(gr_complex) * veclen, samples / veclen});
        auto snk = blocks::null_sink::make({sizeof(gr_complex) * veclen});
        std::vector<blocks::copy::sptr> copy_blks(nblocks);
        for (unsigned int i = 0; i < nblocks; i++) {
            copy_blks[i] = blocks::copy::make({sizeof(gr_complex) * veclen});
        }
        flowgraph_sptr fg(new flowgraph());

        if (buffer_type == 0) {
            fg->connect(src, 0, head, 0);
            fg->connect(head, 0, copy_blks[0], 0);
            for (unsigned int i = 0; i < nblocks - 1; i++) {
                fg->connect(copy_blks[i], 0, copy_blks[i + 1], 0);
            }
            fg->connect(copy_blks[nblocks - 1], 0, snk, 0);

        } else {
            fg->connect(src, 0, head, 0)->set_custom_buffer(VMCIRC_BUFFER_ARGS);
            fg->connect(head, 0, copy_blks[0], 0)->set_custom_buffer(VMCIRC_BUFFER_ARGS);
            for (unsigned int i = 0; i < nblocks - 1; i++) {
                fg->connect(copy_blks[i], 0, copy_blks[i + 1], 0)->set_custom_buffer(VMCIRC_BUFFER_ARGS);
            }
            fg->connect(copy_blks[nblocks - 1], 0, snk, 0)->set_custom_buffer(VMCIRC_BUFFER_ARGS);
        }

        std::cout << "Initializing MT scheduler with buffer size of " << buffer_size << std::endl;
        auto sched = schedulers::scheduler_mt::make("mt", buffer_size);
        fg->add_scheduler(sched);

        if (buffer_type == 1) {
            sched->set_default_buffer_factory(VMCIRC_BUFFER_ARGS);
        }

        if (nthreads > 0) {
            int blks_per_thread = nblocks / nthreads;

            for (unsigned int i = 0; i < nthreads; i++) {
                std::vector<block_sptr> block_group;
                if (i == 0) {
                    block_group.push_back(src);
                    block_group.push_back(head);
                }

                for (int j = 0; j < blks_per_thread; j++) {
                    block_group.push_back(copy_blks[i * blks_per_thread + j]);
                }

                if (i == nthreads - 1) {
                    for (unsigned int j = 0; j < (nblocks - nthreads * blks_per_thread); j++) {
                        block_group.push_back(copy_blks[(i + 1) * blks_per_thread + j]);
                    }
                    block_group.push_back(snk);
                }
                if (cpu_affinity.empty())
                {
                    sched->add_block_group(block_group);
                }
                else
                {
                    sched->add_block_group(block_group, "group" + std::to_string(i), {cpu_affinity[i]} );
                }
                
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
