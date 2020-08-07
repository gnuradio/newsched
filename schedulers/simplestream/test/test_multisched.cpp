#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/domain_adapter_direct.hpp>
#include <gnuradio/domain_adapter_shm.hpp>
#include <gnuradio/domain_adapter_zmq.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    float k1 = 2.0;
    float k2 = 1.0;
    int nsamps = 50000000;
    std::vector<float> input_data(nsamps);
    std::vector<float> expected_output(nsamps);
    for (int i = 0; i < nsamps; i++) {
        input_data[i] = i;
        expected_output[i] = i * k1 * k2;
    }

    if (1) {
        auto src = blocks::vector_source_f::make(input_data, false);
        auto mult1 = blocks::multiply_const_ff::make(k1);
        auto mult2 = blocks::multiply_const_ff::make(k2);
        // auto mult3 = blocks::multiply_const_ff::make(k2);
        // auto mult4 = blocks::multiply_const_ff::make(k2);
        // auto mult5 = blocks::multiply_const_ff::make(k2);
        // auto mult6 = blocks::multiply_const_ff::make(k2);
        // auto mult7 = blocks::multiply_const_ff::make(k2);
        // auto mult8 = blocks::multiply_const_ff::make(k2);
        // auto mult9 = blocks::multiply_const_ff::make(k2);
        // auto mult10 = blocks::multiply_const_ff::make(k2);
        auto snk = blocks::vector_sink_f::make();

        flowgraph_sptr fg(new flowgraph());
        fg->connect(src, 0, mult1, 0);
        fg->connect(mult1, 0, mult2, 0);
        // fg->connect(mult2, 0, mult3, 0);
        // fg->connect(mult3, 0, mult4, 0);
        // fg->connect(mult4, 0, mult5, 0);
        // fg->connect(mult5, 0, mult6, 0);
        // fg->connect(mult6, 0, mult7, 0);
        // fg->connect(mult7, 0, mult8, 0);
        // fg->connect(mult8, 0, mult9, 0);
        // fg->connect(mult9, 0, mult10, 0);
        fg->connect(mult2, 0, snk, 0);
        // fg->connect(mult10, 0, snk, 0);

        std::shared_ptr<schedulers::scheduler_simplestream> sched1(
            new schedulers::scheduler_simplestream("sched1"));
        std::shared_ptr<schedulers::scheduler_simplestream> sched2(
            new schedulers::scheduler_simplestream("sched2"));
        // std::shared_ptr<schedulers::scheduler_simplestream> sched3(
        //     new schedulers::scheduler_simplestream("sched3"));
        // std::shared_ptr<schedulers::scheduler_simplestream> sched4(
        //     new schedulers::scheduler_simplestream("sched4"));
        fg->add_scheduler(sched1);
        fg->add_scheduler(sched2);
        // fg->add_scheduler(sched3);
        // fg->add_scheduler(sched4);

        auto da_conf = domain_adapter_shm_conf::make(buffer_preference_t::UPSTREAM);
        // auto da_conf =
        //     domain_adapter_zmq_tcp_conf::make(std::vector<int>{ 1234, 1235, 1236, 1237 },
        //                                       "127.0.0.1",
        //                                       buffer_preference_t::UPSTREAM);

        domain_conf_vec dconf{ domain_conf(sched1, { src, mult1 }, da_conf),
                               domain_conf(sched2, { mult2, snk }, da_conf) };
        // domain_conf_vec dconf{
        //     domain_conf(sched1, { src, mult1, mult2, mult3, mult4, mult5 }, da_conf),
        //     domain_conf(sched2, { mult6, mult7, mult8, mult9, mult10, snk }, da_conf)
        // };
        // domain_conf_vec dconf{ domain_conf(sched1, { src, mult1, mult2 }, da_conf),
        //                        domain_conf(sched2, { mult3, mult4, mult5 }, da_conf),
        //                        domain_conf(sched3, { mult6, mult7, mult8 }, da_conf),
        //                        domain_conf(sched4, { mult9, mult10, snk }, da_conf) };

        fg->partition(dconf);

        auto t1 = std::chrono::steady_clock::now();
        fg->start();
        fg->wait();

        auto t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        std::cout << "Domain Adapter flowgraph took: " << fp_ms.count() << std::endl;

        auto vec = snk->data();
        std::cout << (vec == expected_output) << " " << vec.size() << " "
                  << expected_output.size() << std::endl;


        for (auto d : vec) {
            // std::cout << d << ",";
        }


        std::cout << std::endl;
    }
    {
        auto src = blocks::vector_source_f::make(input_data, false);
        auto mult1 = blocks::multiply_const_ff::make(k1);
        auto mult2 = blocks::multiply_const_ff::make(k2);
        // auto mult3 = blocks::multiply_const_ff::make(k2);
        // auto mult4 = blocks::multiply_const_ff::make(k2);
        // auto mult5 = blocks::multiply_const_ff::make(k2);
        // auto mult6 = blocks::multiply_const_ff::make(k2);
        // auto mult7 = blocks::multiply_const_ff::make(k2);
        // auto mult8 = blocks::multiply_const_ff::make(k2);
        // auto mult9 = blocks::multiply_const_ff::make(k2);
        // auto mult10 = blocks::multiply_const_ff::make(k2);
        auto snk = blocks::vector_sink_f::make();

        flowgraph_sptr fg(new flowgraph());
        fg->connect(src, 0, mult1, 0);
        fg->connect(mult1, 0, mult2, 0);
        fg->connect(mult2, 0, snk, 0);
        // fg->connect(mult2, 0, mult3, 0);
        // fg->connect(mult3, 0, mult4, 0);
        // fg->connect(mult4, 0, mult5, 0);
        // fg->connect(mult5, 0, mult6, 0);
        // fg->connect(mult6, 0, mult7, 0);
        // fg->connect(mult7, 0, mult8, 0);
        // fg->connect(mult8, 0, mult9, 0);
        // fg->connect(mult9, 0, mult10, 0);
        // fg->connect(mult10, 0, snk, 0);

        std::shared_ptr<schedulers::scheduler_simplestream> sched1(
            new schedulers::scheduler_simplestream("sched1"));

        fg->add_scheduler(sched1);

        fg->validate();

        auto t1 = std::chrono::steady_clock::now();
        fg->start();
        fg->wait();

        auto t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        std::cout << "non-Domain Adapter flowgraph took: " << fp_ms.count() << std::endl;

        auto vec = snk->data();
        std::cout << (vec == expected_output) << std::endl;
        for (auto d : vec) {
            // std::cout << d << ",";
        }

        std::cout << std::endl;
    }
}
