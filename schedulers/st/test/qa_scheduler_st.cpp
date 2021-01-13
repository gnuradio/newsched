#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/st/scheduler_st.hpp>

using namespace gr;

TEST(SchedulerSTTest, TwoSinks)
{
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<float> expected_data;
    int k = 3.0;
    for (auto& inp : input_data)
    {
        expected_data.push_back(inp * k);
    }

    auto src = blocks::vector_source_f::make(input_data, false);
    auto op = blocks::multiply_const_ff::make(k);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();

    auto fg = flowgraph::make();
    fg->connect(src, 0, op, 0);
    fg->connect(op, 0, snk1, 0);
    fg->connect(op, 0, snk2, 0);

    auto sched = schedulers::scheduler_st::make();
    fg->set_scheduler(sched);

    fg->validate();

    fg->start();
    fg->wait();

    EXPECT_EQ(snk1->data(), expected_data);
    EXPECT_EQ(snk2->data(), expected_data);
}

