#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocks/multiply_const.hpp>
#include <gnuradio/blocks/vector_sink.hpp>
#include <gnuradio/blocks/vector_source.hpp>
#include <gnuradio/buffer_sm.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/mt/scheduler_mt.hpp>

using namespace gr;

#if 0
TEST(SchedulerMTSingleBuffers, SingleMappedSimple)
{
    int nsamples = 1000000;
    std::vector<gr_complex> input_data(nsamples);
    std::vector<gr_complex> expected_data(nsamples);

    float k = 1.0;
    for (int i = 0; i < nsamples; i++) {
        input_data[i] = gr_complex(2 * i, 2 * i + 1);
    }

    int nblocks = 2;
    int veclen = 1;
    auto src = blocks::vector_source_c::make(input_data);
    auto snk = blocks::vector_sink_c::make();
    std::vector<blocks::multiply_const_cc::sptr> mult_blks(nblocks);

    for (int i = 0; i < nblocks; i++) {
        mult_blks[i] = blocks::multiply_const_cc::make(k, veclen);
    }

    auto fg = flowgraph::make();

    fg->connect(src, 0, mult_blks[0], 0);
    for (int i = 1; i < nblocks; i++) {
        fg->connect(mult_blks[i - 1], 0, mult_blks[i], 0)
            ->set_custom_buffer(SM_BUFFER_ARGS);
    }
    fg->connect(mult_blks[nblocks - 1], 0, snk, 0);

    auto sched1 = schedulers::scheduler_mt::make("mtsched", 8192);
    fg->add_scheduler(sched1);
    fg->validate();


    fg->start();
    fg->wait();

    for (int i = 0; i < nsamples; i++) {
        input_data[i] = gr_complex(2 * i, 2 * i + 1);
        expected_data[i] = gr_complex(k * 2 * i, k * (2 * i + 1));
    }


    EXPECT_EQ(snk->data(), expected_data);
    EXPECT_EQ(snk->data().size(), expected_data.size());
}
#endif

// Test the case where we have multiple readers to a single block
TEST(SchedulerMTSingleBuffers, SingleMappedFanout)
{
    int nsamples = 1000000;
    std::vector<gr_complex> input_data(nsamples);
    std::vector<gr_complex> expected_data(nsamples);

    float k = 1.0;
    for (int i = 0; i < nsamples; i++) {
        input_data[i] = gr_complex(2 * i, 2 * i + 1);
    }

    auto nblocks = 4;


    int veclen = 1;
    auto src = blocks::vector_source_c::make(input_data);
    std::vector<blocks::vector_sink_c::sptr> sink_blks(nblocks);
    std::vector<blocks::multiply_const_cc::sptr> mult_blks(nblocks);

    for (int i = 0; i < nblocks; i++) {
        mult_blks[i] = blocks::multiply_const_cc::make(k, veclen);
        sink_blks[i] = blocks::vector_sink_c::make();
    }
    flowgraph_sptr fg(new flowgraph());

    for (int i = 0; i < nblocks; i++) {
        fg->connect(src, 0, mult_blks[i], 0)->set_custom_buffer(SM_BUFFER_ARGS);
        fg->connect(mult_blks[i], 0, sink_blks[i], 0)->set_custom_buffer(SM_BUFFER_ARGS);
    }

    auto sched1 = schedulers::scheduler_mt::make("mtsched", 8192);
    fg->add_scheduler(sched1);
    fg->validate();


    fg->start();
    fg->wait();

    for (int n = 0; n < nblocks; n++) {
        for (int i = 0; i < nsamples; i++) {
            input_data[i] = gr_complex(2 * i, 2 * i + 1);
            expected_data[i] = gr_complex(k * 2 * i, k * (2 * i + 1));
        }

        EXPECT_EQ(sink_blks[n]->data(), expected_data);
        EXPECT_EQ(sink_blks[n]->data().size(), expected_data.size());
    }
}
