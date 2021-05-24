#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocks/vector_sink.hh>
#include <gnuradio/blocks/vector_source.hh>
// #include <gnuradio/cuda/copy.hh>
#include <gnuradio/blocks/copy.hh>
#include <gnuradio/cudabuffer.hh>
#include <gnuradio/cudabuffer_pinned.hh>
#include <gnuradio/flowgraph.hh>
#include <gnuradio/schedulers/mt/scheduler_mt.hh>

using namespace gr;

/*
 * Test a basic chain of copy blocks, all assigned to the same thread
 *
 */
TEST(SchedulerMTTest, CudaCopyBasic)
{
    size_t veclen = 1024;
    int num_samples = veclen*100;
    std::vector<gr_complex> input_data(num_samples);

    for (int i = 0; i < num_samples; i++) {
        input_data[i] = gr_complex(i, -i);
    }

    auto src = blocks::vector_source_c::make_cpu({input_data, false, veclen});
    auto snk1 = blocks::vector_sink_c::make_cpu({veclen});
    auto copy1 = blocks::copy::make_cuda({veclen*sizeof(gr_complex)});
    auto copy2 = blocks::copy::make_cuda({veclen*sizeof(gr_complex)});

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

    EXPECT_EQ(snk1->data().size(), input_data.size());
    EXPECT_EQ(snk1->data(), input_data);
}

/*
 * Test a basic chain of copy blocks, on different threads
 *
 */
TEST(SchedulerMTTest, CudaCopyMultiThreaded)
{
    size_t veclen = 1024;
    int num_samples = veclen*1000;
    std::vector<gr_complex> input_data(num_samples);

    for (int i = 0; i < num_samples; i++) {
        input_data[i] = gr_complex(i, -i);
    }

    auto src = blocks::vector_source_c::make_cpu({input_data, false, veclen});
    auto snk1 = blocks::vector_sink_c::make_cpu({veclen});
    auto copy1 = blocks::copy::make_cuda({veclen*sizeof(gr_complex)});
    auto copy2 = blocks::copy::make_cuda({veclen*sizeof(gr_complex)});

    auto fg = flowgraph::make();
    // fg->connect(src, 0, copy1, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D);
    // fg->connect(copy1, 0, copy2, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D);
    // fg->connect(copy2, 0, snk1, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2H);
    fg->connect(src, 0, copy1, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    fg->connect(copy1, 0, copy2, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);
    fg->connect(copy2, 0, snk1, 0)->set_custom_buffer(CUDA_BUFFER_PINNED_ARGS);

    auto sched = schedulers::scheduler_mt::make("sched", 32768);
    fg->set_scheduler(sched);
    // by not adding block group, each block in its own thread

    fg->validate();

    fg->start();
    fg->wait();

    auto outdata = snk1->data();

    EXPECT_EQ(outdata.size(), input_data.size());
    EXPECT_EQ(outdata, input_data);

    // for (int i=0; i<outdata.size(); i++)
    // {
    //     if (outdata[i] != input_data[i])
    //     std::cout << outdata[i] << " " << input_data[i] << std::endl;
    // }
    
}
