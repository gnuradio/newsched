#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/blocklib/cuda/copy.hpp>
#include <gnuradio/cudabuffer.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/mt/scheduler_mt.hpp>

using namespace gr;

/*
 * Test a basic chain of copy blocks, all assigned to the same thread
 *
 */
TEST(SchedulerMTTest, CudaCopyBasic)
{
    int num_samples = 100000;
    std::vector<gr_complex> input_data(num_samples);

    for (int i = 0; i < num_samples; i++) {
        input_data[i] = gr_complex(i, -i);
    }

    auto src = blocks::vector_source_c::make(input_data, false);
    auto snk1 = blocks::vector_sink_c::make();
    auto copy1 = cuda::copy::make(1024);
    auto copy2 = cuda::copy::make(1024);

    auto fg = flowgraph::make();
    fg->connect(src, 0, copy1, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D);
    fg->connect(copy1, 0, copy2, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D);
    fg->connect(copy2, 0, snk1, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2H);

    auto sched = schedulers::scheduler_mt::make("sched", 32768);
    fg->set_scheduler(sched);
    sched->add_block_group({ copy1, copy2 });

    fg->validate();

    fg->start();
    fg->wait();

    EXPECT_EQ(snk1->data(), input_data);
}

/*
 * Test a basic chain of copy blocks, on different threads
 *
 */
TEST(SchedulerMTTest, CudaCopyMultiThreaded)
{
    int num_samples = 100000;
    std::vector<gr_complex> input_data(num_samples);

    for (int i = 0; i < num_samples; i++) {
        input_data[i] = gr_complex(i, -i);
    }

    auto src = blocks::vector_source_c::make(input_data, false);
    auto snk1 = blocks::vector_sink_c::make();
    auto copy1 = cuda::copy::make(1024);
    auto copy2 = cuda::copy::make(1024);

    auto fg = flowgraph::make();
    fg->connect(src, 0, copy1, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D);
    fg->connect(copy1, 0, copy2, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D);
    fg->connect(copy2, 0, snk1, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2H);

    auto sched = schedulers::scheduler_mt::make("sched", 32768);
    fg->set_scheduler(sched);
    // by not adding block group, each block in its own thread

    fg->validate();

    fg->start();
    fg->wait();

    EXPECT_EQ(snk1->data(), input_data);
}
