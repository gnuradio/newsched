#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/fanout.hpp>
#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/domain.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    auto src = blocks::vector_source_f::make(
        std::vector<float>{ 1.0, 2.0, 3.0, 4.0, 5.0 }, false);
    auto throttle = blocks::throttle::make(sizeof(float), 100);
    auto fanout = blocks::fanout_cc::make(2);
    auto mult1 = blocks::multiply_const_ff::make(100.0);
    auto mult2 = blocks::multiply_const_ff::make(200.0);
    auto mult3 = blocks::multiply_const_ff::make(300.0);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, throttle, 0);
    fg->connect(throttle, 0, mult1, 0);
    fg->connect(mult1, 0, fanout, 0);
    fg->connect(fanout, 0, mult2, 0);
    fg->connect(fanout, 1, mult3, 0);
    fg->connect(mult2, 0, snk1, 0);
    fg->connect(mult3, 0, snk2, 0);

    std::shared_ptr<schedulers::scheduler_simplestream> sched1(
        new schedulers::scheduler_simplestream("sched1"));
    std::shared_ptr<schedulers::scheduler_simplestream> sched2(
        new schedulers::scheduler_simplestream("sched2"));
    std::shared_ptr<schedulers::scheduler_simplestream> sched3(
        new schedulers::scheduler_simplestream("sched3"));

    fg->add_scheduler(sched1);
    fg->add_scheduler(sched2);
    fg->add_scheduler(sched3);

    // partition_conf_vec partitions{ { sched1, { src, throttle, mult1 } },
    //                                { sched2, { mult2, snk1 } },
    //                                { sched3, { mult3, snk2 } } };


#if 0
    domain_conf_vec dconf{ domain_conf(sched1, { src, throttle, mult1, fanout}),
                           domain_conf(sched2,
                                       { mult2, snk1 },
                                       domain_adapter_zmq_conf::make(
                                           buffer_preference_t::UPSTREAM,
                                           "tcp://127.0.0.1:1234",
                                           "tcp://127.0.0.1:1234")),
                           domain_conf(sched3, { mult3, snk2 },
                                       domain_adapter_zmq_conf::make(
                                           buffer_preference_t::UPSTREAM,
                                           "tcp://127.0.0.1:1235",
                                           "tcp://127.0.0.1:1235")) };
#else
    auto da_conf =
        domain_adapter_zmq_tcp_conf::make(std::vector<int>{ 1234, 1235, 1236, 1237 }, "127.0.0.1", buffer_preference_t::UPSTREAM);

    domain_conf_vec dconf{ domain_conf(sched1, { src, throttle, mult1, fanout }),
                           domain_conf(sched2, { mult2, snk1 }, da_conf),
                           domain_conf(sched3, { mult3, snk2 }, da_conf) };
#endif

    // domain_conf_vec dconf{ domain_conf(sched1, { src, throttle, mult1 }),
    //                        domain_conf(sched2, { mult2, snk1 }),
    //                        domain_conf(sched3, { mult3, snk2 }) };


    fg->partition(dconf);

    fg->start();
    fg->wait();

    // TODO: Logging module
    for (const auto& d : snk1->data())
        std::cout << d << ' ';
    std::cout << std::endl;

    for (const auto& d : snk2->data())
        std::cout << d << ' ';
    std::cout << std::endl;
}
