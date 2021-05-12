
#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocks/copy.hpp>
#include <gnuradio/blocks/multiply_const.hpp>
#include <gnuradio/blocks/vector_sink.hpp>
#include <gnuradio/blocks/vector_source.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/mt/scheduler_mt.hpp>
#include <gnuradio/simplebuffer.hpp>

using namespace gr;


TEST(SchedulerMTTest, TwoSinks)
{
    int nsamples = 100000;
    std::vector<float> input_data(nsamples);
    for (int i = 0; i < nsamples; i++) {
        input_data[i] = i;
    }
    auto src = blocks::vector_source_f::make(input_data, false);
    auto mult1 = blocks::multiply_const_ff::make(1.0);
    auto mult2 = blocks::multiply_const_ff::make(1.0);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();


    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, mult1, 0)->set_custom_buffer(VMCIRC_BUFFER_MMAP_SHM_ARGS);
    fg->connect(mult1, 0, snk1, 0)->set_custom_buffer(VMCIRC_BUFFER_MMAP_SHM_ARGS);
    fg->connect(src, 0, mult2, 0)->set_custom_buffer(VMCIRC_BUFFER_MMAP_SHM_ARGS);
    fg->connect(mult2, 0, snk2, 0)->set_custom_buffer(VMCIRC_BUFFER_MMAP_SHM_ARGS);

    auto sched = schedulers::scheduler_mt::make();
    fg->set_scheduler(sched);

    // force single threaded operation
    // sched->add_block_group({src,snk1,snk2});

    fg->validate();

    fg->start();
    fg->wait();

    EXPECT_EQ(snk1->data().size(), input_data.size());
    EXPECT_EQ(snk2->data().size(), input_data.size());
    EXPECT_EQ(snk1->data(), input_data);
    EXPECT_EQ(snk2->data(), input_data);
}
