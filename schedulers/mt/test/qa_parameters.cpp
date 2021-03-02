#include <gtest/gtest.h>

#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/domain_adapter_direct.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/mt/scheduler_mt.hpp>

using namespace gr;

TEST(SchedulerSTTest, ParameterBasic)
{
    auto src = blocks::vector_source_f::make(
        std::vector<float>{ 1.0, 2.0, 3.0, 4.0, 5.0 }, true);
    auto throttle = blocks::throttle::make(sizeof(float), 32000);
    auto mult1 = blocks::multiply_const_ff::make(100.0);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, throttle, 0);
    fg->connect(throttle, 0, mult1, 0);
    fg->connect(mult1, 0, snk1, 0);
    fg->connect(mult1, 0, snk2, 0);

    auto sched = schedulers::scheduler_mt::make("sched1", 4096);
    fg->set_scheduler(sched);

    fg->validate();
    fg->start();

    auto start = std::chrono::steady_clock::now();

    float k = 1.0;

    std::vector<float> k_set;
    std::vector<float> k_queried;

    while (true) {
        mult1->set_k(k);
        k_set.push_back(k);

        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(1))
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        k += 1.0;

        auto query_k = mult1->k();
        // std::cout << query_k << std::endl;
        k_queried.push_back(query_k);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    fg->stop();

    // Search for the values that we set in the values we queried
    bool all_values_found = true;
    for (unsigned i = 0; i<k_set.size()-1; i++)
    {
        if (std::find(k_queried.begin(), k_queried.end(), k_set[i]) == k_queried.end())
        {
            all_values_found = false;
        }
    }

    EXPECT_TRUE(all_values_found);

    // now look at the data
    EXPECT_GT(snk1->data().size(), 5);
    EXPECT_GT(snk2->data().size(), 5);

    // TODO - check for query
    // TODO - set changes at specific sample numbers
}
