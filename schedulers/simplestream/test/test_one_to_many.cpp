#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/fanout.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/multiply_const_blk.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    auto src = blocks::vector_source_f::make(
        std::vector<float>{ 1.0, 2.0, 3.0, 4.0, 5.0 }, true);
    auto throttle = blocks::throttle::make(sizeof(float), 32000);
    auto mult1 = blocks::multiply_const_ff::make(100.0);
    auto fanout = blocks::fanout_ff::make(2);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, throttle, 0);
    fg->connect(throttle, 0, mult1, 0);
    fg->connect(mult1, 0, fanout, 0);
    fg->connect(fanout, 0, snk1, 0);
    fg->connect(fanout, 1, snk2, 0);

    std::shared_ptr<schedulers::scheduler_simplestream> sched(
        new schedulers::scheduler_simplestream());
    fg->set_scheduler(sched);

    fg->validate();
    fg->start();

    auto start = std::chrono::steady_clock::now();

    float k = 1.0;

    while (true) {
        auto query_k = mult1->k();

        mult1->set_k(k);

        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5))
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        k += 1.0;
    }

    fg->stop();

    // for (const auto& d : snk1->data())
    //     std::cout << d << ' ';
    // std::cout << std::endl;

    // for (const auto& d : snk2->data())
    //     std::cout << d << ' ';
    // std::cout << std::endl;

}