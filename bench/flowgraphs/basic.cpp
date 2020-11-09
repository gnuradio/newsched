#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/null_source.hpp>
#include <gnuradio/blocklib/blocks/head.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/domain_adapter_shm.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/st/scheduler_st.hpp>
#include <gnuradio/realtime.hpp>
#include <iostream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace gr;

int main(int argc, char* argv[])
{
    int run;
    uint64_t samples;
    bool rt_prio = false;
    bool machine_readable = false;

    po::options_description desc("Basic Test Flow Graph");
    desc.add_options()
        ("help,h", "display help")
        ("run,R", po::value<int>(&run)->default_value(0), "Run Number")
        ("samples,N", po::value<uint64_t>(&samples)->default_value(15000000), "Number of samples")
        ("machine_readable,m", "Machine-readable Output")
        ("rt_prio,t", "Enable Real-time priority");
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

    float k = 1.0;

    {
        // auto src = blocks::vector_source_f::make(input_data, false);
        auto src = blocks::null_source::make(sizeof(float));
        auto head = blocks::head::make(sizeof(float), samples);
        auto mult = blocks::multiply_const_ff::make(k);
        auto snk = blocks::vector_sink_f::make();

        flowgraph_sptr fg(new flowgraph());
        fg->connect(src, 0, head, 0);
        fg->connect(head, 0, mult, 0);
        fg->connect(mult, 0, snk, 0);

        std::shared_ptr<schedulers::scheduler_st> sched1(
            new schedulers::scheduler_st("sched1"));

        fg->add_scheduler(sched1);

        fg->validate();

        if(gr::enable_realtime_scheduling() != gr::rt_status_t::RT_OK)
            std::cout << "Unable to enable realtime scheduling " << std::endl;
        
        auto t1 = std::chrono::steady_clock::now();

        fg->start();
        fg->wait();

        auto t2 = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()/1e9;

        std::cout << "[PROFILE]"  << time << "[PROFILE]" << std::endl;

        // std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        // std::cout << "non-Domain Adapter flowgraph took: " << fp_ms.count() << std::endl;

    }
}
