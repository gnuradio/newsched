#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/multiply_const_blk.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    auto src = blocks::vector_source_f::make(
        std::vector<float>{ 1.0, 2.0, 3.0, 4.0, 5.0 }, false);
    auto throttle = blocks::throttle::make(sizeof(float), 100);
    auto mult1 = blocks::multiply_const_ff::make(100.0);
    auto mult2 = blocks::multiply_const_ff::make(200.0);
    auto snk = blocks::vector_sink_f::make();

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, throttle, 0);
    fg->connect(throttle, 0, mult1, 0);
    fg->connect(mult1, 0, mult2, 0);
    fg->connect(mult2, 0, snk, 0);

    std::shared_ptr<schedulers::scheduler_simplestream> sched1(
        new schedulers::scheduler_simplestream("sched1"));
    std::shared_ptr<schedulers::scheduler_simplestream> sched2(
        new schedulers::scheduler_simplestream("sched2"));

    fg->add_scheduler(sched1);
    fg->add_scheduler(sched2);

    partition_conf_vec partitions{ { sched1, { src, throttle, mult1 } },
                                   { sched2, { mult2, snk } } };

    fg->partition(partitions);

    fg->start();
    fg->wait();

    for (const auto& d : snk->data())
        std::cout << d << ' ';
    std::cout << std::endl;
}