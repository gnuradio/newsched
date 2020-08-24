// #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include <doctest.h>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/fanout.hpp>
#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>
#include <gnuradio/domain_adapter_shm.hpp>

using namespace gr;


TEST_CASE("block outputs one output to 2 input blocks")
{
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    auto src = blocks::vector_source_f::make(input_data, false);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();


    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, snk1, 0);
    fg->connect(src, 0, snk2, 0);

    std::shared_ptr<schedulers::scheduler_simplestream> sched(
        new schedulers::scheduler_simplestream());
    fg->set_scheduler(sched);

    fg->validate();

    fg->start();
    fg->wait();

    REQUIRE_THAT(snk1->data(), Catch::Equals(input_data));
    REQUIRE_THAT(snk2->data(), Catch::Equals(input_data));
}

#if 1
TEST_CASE("Two schedulers connected by domain adapters internally")
{
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    std::vector<float> expected_data;
    for (auto d : input_data) {
        expected_data.push_back(100.0 * 200.0 * d);
    }

    auto src = blocks::vector_source_f::make(input_data, false);
    auto throttle = blocks::throttle::make(sizeof(float), 100);
    auto mult1 = blocks::multiply_const_ff::make(100.0);
    auto mult2 = blocks::multiply_const_ff::make(200.0);
    auto snk = blocks::vector_sink_f::make();

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, mult1, 0);
    fg->connect(mult1, 0, mult2, 0);
    fg->connect(mult2, 0, snk, 0);

    std::shared_ptr<schedulers::scheduler_simplestream> sched1(
        new schedulers::scheduler_simplestream("sched1"));
    std::shared_ptr<schedulers::scheduler_simplestream> sched2(
        new schedulers::scheduler_simplestream("sched2"));

    fg->add_scheduler(sched1);
    fg->add_scheduler(sched2);

    auto da_conf =
        domain_adapter_shm_conf::make(buffer_preference_t::UPSTREAM);

    domain_conf_vec dconf{ domain_conf(sched1, { src, mult1 }, da_conf),
                           domain_conf(sched2, { mult2, snk }, da_conf) };

    fg->partition(dconf);

    fg->start();
    fg->wait();

    REQUIRE_THAT(snk->data(), Catch::Equals(expected_data));
}
#endif

#if 1
TEST_CASE("2 sinks, query and set parameters while FG is running")
{
    auto src = blocks::vector_source_f::make(
        std::vector<float>{ 1.0, 2.0, 3.0, 4.0, 5.0 }, true);
    auto throttle = blocks::throttle::make(sizeof(float), 32000);
    auto mult1 = blocks::multiply_const_ff::make(100.0);
    auto fanout = blocks::fanout::make(sizeof(float), 2);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, throttle, 0);
    fg->connect(throttle, 0, mult1, 0);
    fg->connect(mult1, 0, fanout, 0);
    fg->connect(fanout, 0, snk1, 0);
    fg->connect(fanout, 1, snk2, 0);

    std::shared_ptr<schedulers::scheduler_simplestream> sched(
        new schedulers::scheduler_simplestream("sched1",100));
    fg->set_scheduler(sched);

    fg->validate();

    fg->start();

    auto start = std::chrono::steady_clock::now();

    float k = 1.0;

    while (true) {
        auto query_k = mult1->k();

        mult1->set_k(k);

        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(1))
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        k += 1.0;
    }

    fg->stop();

    std::cout << "fg stopped" << std::endl;

    // now look at the data
    REQUIRE(snk1->data().size() > 5);
    REQUIRE(snk2->data().size() > 5);

    // TODO - check for query
    // TODO - set changes at specific sample numbers
}
#endif
