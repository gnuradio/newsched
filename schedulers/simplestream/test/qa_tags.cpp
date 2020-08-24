// #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include <doctest.h>
// #define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/fanout.hpp>
#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/domain_adapter_shm.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>

using namespace gr;


TEST_CASE("Default ALL_TO_ALL propagation")
{
    float k1 = 2.0;
    int nsamps = 10000;
    std::vector<float> input_data(nsamps);
    std::vector<float> expected_output(nsamps);
    for (int i = 0; i < nsamps; i++) {
        input_data[i] = i;
        expected_output[i] = i * k1;
    }


    std::vector<tag_t> input_tags{
        tag_t(17, "key1", "value1"),   tag_t(39, "key2", "value2"),
        tag_t(84, "key3", "value3"),   tag_t(9017, "key4", "value4"),
        tag_t(9039, "key5", "value5"), tag_t(9084, "key6", "value6")
    };
    auto src = blocks::vector_source_f::make(input_data, false, 1, input_tags);
    auto mult = blocks::multiply_const_ff::make(k1);
    auto snk = blocks::vector_sink_f::make();

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, mult, 0);
    fg->connect(mult, 0, snk, 0);


    std::shared_ptr<schedulers::scheduler_simplestream> sched1(
        new schedulers::scheduler_simplestream("sched1"));

    fg->add_scheduler(sched1);

    fg->validate();

    fg->start();
    fg->wait();


    auto vec = snk->data();
    std::cout << (vec == expected_output) << " " << vec.size() << " "
              << expected_output.size() << std::endl;

    auto out_tags = snk->tags();

    REQUIRE_THAT(snk->data(), Catch::Equals(expected_output));
    REQUIRE(snk->tags() == input_tags);
}


TEST_CASE("Default ALL_TO_ALL propagation with fanout")
{
    float k1 = 2.0;
    int nsamps = 10000;
    std::vector<float> input_data(nsamps);
    std::vector<float> expected_output(nsamps);
    for (int i = 0; i < nsamps; i++) {
        input_data[i] = i;
        expected_output[i] = i * k1;
    }


    std::vector<tag_t> input_tags{
        tag_t(17, "key1", "value1"),   tag_t(39, "key2", "value2"),
        tag_t(84, "key3", "value3"),   tag_t(9017, "key4", "value4"),
        tag_t(9039, "key5", "value5"), tag_t(9084, "key6", "value6")
    };
    auto src = blocks::vector_source_f::make(input_data, false, 1, input_tags);
    auto mult = blocks::multiply_const_ff::make(k1);
    auto fan = blocks::fanout::make(sizeof(float), 4);

    std::vector<blocks::vector_sink_f::sptr> snks;


    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, mult, 0);
    fg->connect(mult, 0, fan, 0);
    for (int i = 0; i < 4; i++) {
        snks.push_back(blocks::vector_sink_f::make());
        fg->connect(fan, i, snks[i], 0);
    }


    std::shared_ptr<schedulers::scheduler_simplestream> sched1(
        new schedulers::scheduler_simplestream("sched1"));

    fg->add_scheduler(sched1);

    fg->validate();

    fg->start();
    fg->wait();


    for (auto snk : snks) {
        REQUIRE_THAT(snk->data(), Catch::Equals(expected_output));
        REQUIRE(snk->tags() == input_tags);
    }
}

TEST_CASE("ONE_TO_ONE propagation with fanout")
{
    float k1 = 2.0;
    int nsamps = 10000;
    std::vector<float> input_data(nsamps);
    std::vector<float> expected_output(nsamps);
    for (int i = 0; i < nsamps; i++) {
        input_data[i] = i;
        expected_output[i] = i * k1;
    }


    std::vector<tag_t> input_tags{
        tag_t(17, "key1", "value1"),   tag_t(39, "key2", "value2"),
        tag_t(84, "key3", "value3"),   tag_t(9017, "key4", "value4"),
        tag_t(9039, "key5", "value5"), tag_t(9084, "key6", "value6")
    };
    auto src = blocks::vector_source_f::make(input_data, false, 1, input_tags);
    auto mult = blocks::multiply_const_ff::make(k1);
    auto fan = blocks::fanout::make(sizeof(float), 4);
    fan->set_tag_propagation_policy(tag_propagation_policy_t::TPP_ONE_TO_ONE);

    std::vector<blocks::vector_sink_f::sptr> snks;


    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, mult, 0);
    fg->connect(mult, 0, fan, 0);
    for (int i = 0; i < 4; i++) {
        snks.push_back(blocks::vector_sink_f::make());
        fg->connect(fan, i, snks[i], 0);
    }


    std::shared_ptr<schedulers::scheduler_simplestream> sched1(
        new schedulers::scheduler_simplestream("sched1"));

    fg->add_scheduler(sched1);

    fg->validate();

    fg->start();
    fg->wait();

    int idx = 0;
    for (auto snk : snks) {
        REQUIRE_THAT(snk->data(), Catch::Equals(expected_output));
        if (idx == 0) {
            REQUIRE(snk->tags() == input_tags);
        } else {
            REQUIRE(snk->tags().size() == 0);
        }
        idx++;
    }
}