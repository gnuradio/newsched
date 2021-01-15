#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/domain_adapter_direct.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/mt/scheduler_mt.hpp>

using namespace gr;

TEST(SchedulerMTTest, TwoSinks)
{
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    auto src = blocks::vector_source_f::make(input_data, false);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();


    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, snk1, 0);
    fg->connect(src, 0, snk2, 0);

    std::shared_ptr<schedulers::scheduler_mt> sched(new schedulers::scheduler_mt());
    fg->set_scheduler(sched);

    fg->validate();

    fg->start();
    fg->wait();

    EXPECT_EQ(snk1->data(), input_data);
    EXPECT_EQ(snk2->data(), input_data);
}

TEST(SchedulerMTTest, DomainAdapterBasic)
{
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    std::vector<float> expected_data;
    for (auto d : input_data) {
        expected_data.push_back(100.0 * 200.0 * d);
    }

    auto src = blocks::vector_source_f::make(input_data, false);
    auto mult1 = blocks::multiply_const_ff::make(100.0);
    auto mult2 = blocks::multiply_const_ff::make(200.0);
    auto snk = blocks::vector_sink_f::make();

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, mult1, 0);
    fg->connect(mult1, 0, mult2, 0);
    fg->connect(mult2, 0, snk, 0);

    auto sched1 = schedulers::scheduler_mt::make("sched1");
    auto sched2 = schedulers::scheduler_mt::make("sched2");

    fg->add_scheduler(sched1);
    fg->add_scheduler(sched2);

    auto da_conf = domain_adapter_direct_conf::make(buffer_preference_t::UPSTREAM);

    domain_conf_vec dconf{ domain_conf(sched1, { src, mult1 }, da_conf),
                           domain_conf(sched2, { mult2, snk }, da_conf) };

    fg->partition(dconf);

    fg->start();
    fg->wait();

    EXPECT_EQ(snk->data(), expected_data);
}
