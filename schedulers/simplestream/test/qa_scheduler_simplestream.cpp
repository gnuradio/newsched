// #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include <doctest.h>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/dummy.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/blocklib/blocks/multiply_const_blk.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>

using namespace gr;


TEST_CASE("block outputs one output to 2 input blocks")
{
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    auto src = blocks::vector_source_f::make(input_data, false);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();


    flowgraph_sptr fg(new flowgraph());
    fg->connect(src->base(), 0, snk1->base(), 0);
    fg->connect(src->base(), 0, snk2->base(), 0);

    std::shared_ptr<schedulers::scheduler_simplestream> sched(
        new schedulers::scheduler_simplestream());
    fg->set_scheduler(sched->base());

    fg->validate();

    fg->start();
    fg->wait();

    REQUIRE_THAT(snk1->data(), Catch::Equals(input_data));
    REQUIRE_THAT(snk2->data(), Catch::Equals(input_data));
}


TEST_CASE("Two schedulers connected by domain adapters internally")
{
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    std::vector<float> expected_data;
    for (auto d : input_data)
    {
        expected_data.push_back(100.0*200.0*d);
    }

    auto src = blocks::vector_source_f::make(
        input_data, false);
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

    REQUIRE_THAT(snk->data(), Catch::Equals(expected_data));
}

TEST_CASE("2 sinks, query and set parameters while FG is running")
{
    auto src = blocks::vector_source_f::make(
        std::vector<float>{ 1.0, 2.0, 3.0, 4.0, 5.0 }, true);
    auto throttle = blocks::throttle::make(sizeof(float), 32000);
    auto dummy = blocks::dummy<float>::make(7.0, 13.0);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src->base(), 0, throttle->base(), 0);
    fg->connect(throttle->base(), 0, dummy->base(), 0);
    fg->connect(dummy->base(), 0, snk1->base(), 0);
    fg->connect(dummy->base(), 1, snk2->base(), 0);

    std::shared_ptr<schedulers::scheduler_simplestream> sched(
        new schedulers::scheduler_simplestream());
    fg->set_scheduler(sched->base());

    fg->validate();
    fg->start();

    auto start = std::chrono::steady_clock::now();

    float a = 1.0;
    float b = 100.0;

    while (true) {
        auto query_a = dummy->a();
        auto query_b = dummy->b();

        dummy->set_a(a);
        dummy->set_b(b);

        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(1))
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        a += 1.0;
        b += 1.0;
    }

    fg->stop();

    // now look at the data
    REQUIRE(snk1->data().size() > 5);
    REQUIRE(snk2->data().size() > 5);

    // TODO - check for query
    // TODO - set changes at specific sample numbers
}