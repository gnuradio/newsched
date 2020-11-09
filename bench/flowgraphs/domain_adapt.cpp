#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/head.hpp>
#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/null_sink.hpp>
#include <gnuradio/blocklib/blocks/null_source.hpp>
#include <gnuradio/domain_adapter_shm.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/realtime.hpp>
#include <gnuradio/schedulers/st/scheduler_st.hpp>
#include <iostream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace gr;

int main(int argc, char* argv[])
{
    int run;
    uint64_t samples;
    int nsched;
    int nblocks;
    bool rt_prio = false;
    bool machine_readable = false;

    po::options_description desc("Basic Test Flow Graph");
    desc.add_options()("help,h", "display help")(
        "run,R", po::value<int>(&run)->default_value(0), "Run Number")(
        "samples,N",
        po::value<uint64_t>(&samples)->default_value(15000000),
        "Number of samples")(
        "nsched,s", po::value<int>(&nsched)->default_value(2), "Number of schedulers")(
        "nblocks,b", po::value<int>(&nblocks)->default_value(4), "Total mult blocks")(
        "machine_readable,m", "Machine-readable Output")("rt_prio,t",
                                                         "Enable Real-time priority");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    if (vm.count("machine_readable")) {
        machine_readable = true;
    }

    if (vm.count("rt_prio")) {
        rt_prio = true;
    }

    if (rt_prio && gr::enable_realtime_scheduling() != RT_OK) {
        std::cout << "Error: failed to enable real-time scheduling." << std::endl;
    }

    domain_conf_vec dconf;
    auto da_conf = domain_adapter_shm_conf::make(buffer_preference_t::UPSTREAM);

    float k = 1.0;
    int blocks_per_sched = nblocks / nsched;

    {
        // auto src = blocks::vector_source_f::make(input_data, false);
        auto src = blocks::null_source::make(sizeof(float));
        auto head = blocks::head::make(sizeof(float), samples);
        auto snk = blocks::null_sink::make(sizeof(float));

        flowgraph_sptr fg(new flowgraph());

        block_sptr last_block = src;

        for (int t = 0; t < nsched; t++) {

            std::vector<block_sptr> blocks_in_here;
            for (int b = 0; b < blocks_per_sched; b++) {
                auto this_block = blocks::multiply_const_ff::make(k);

                fg->connect(last_block, 0, this_block, 0);
                last_block = this_block;

                blocks_in_here.push_back(this_block);
            }

            std::shared_ptr<schedulers::scheduler_st> sched(
                new schedulers::scheduler_st("sched" + std::to_string(t)));

            fg->add_scheduler(sched);

            if (t == 0) {
                blocks_in_here.push_back(src);
            }
            if (t == nsched - 1) {
                fg->connect(last_block, 0, head, 0);
                fg->connect(head, 0, snk, 0);
                blocks_in_here.push_back(head);
                blocks_in_here.push_back(snk);
            }

            dconf.push_back({ sched, blocks_in_here, da_conf });
        }

        fg->partition(dconf);

        if (rt_prio && gr::enable_realtime_scheduling() != gr::rt_status_t::RT_OK)
            std::cout << "Unable to enable realtime scheduling " << std::endl;

        auto t1 = std::chrono::steady_clock::now();
        fg->start();
        fg->wait();

        auto t2 = std::chrono::steady_clock::now();
        auto time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9;

        std::cout << "[PROFILE]" << time << "[PROFILE]" << std::endl;
    }
}
