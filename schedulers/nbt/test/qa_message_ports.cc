#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocks/msg_forward.hh>
#include <gnuradio/flowgraph.hh>
#include <gnuradio/schedulers/nbt/scheduler_nbt.hh>

#include <thread>
#include <chrono>

using namespace gr;

TEST(SchedulerMTMessagePassing, Forward)
{
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    auto blk1 = blocks::msg_forward::make({});
    auto blk2 = blocks::msg_forward::make({});
    auto blk3 = blocks::msg_forward::make({});

    flowgraph_sptr fg(new flowgraph());
    fg->connect(blk1, "out", blk2, "in");
    fg->connect(blk2, "out", blk3, "in");

    std::shared_ptr<schedulers::scheduler_nbt> sched(new schedulers::scheduler_nbt());
    fg->set_scheduler(sched);

    fg->validate();

    auto src_port = blk1->get_message_port("out");
    for (int i=0; i<10; i++)
    {
        src_port->post(pmt::intern("message"));
    }

    fg->start();

    // auto start = std::chrono::steady_clock::now();
  

    size_t cnt = 0;
    int num_iters = 0;
    while(true)
    {
        cnt = blk3->message_count();
        // auto end = std::chrono::steady_clock::now();
        if (cnt >= 10)
        {
            break;
        }
        std::this_thread::sleep_for (std::chrono::seconds(1));
        num_iters++;
        if (num_iters >= 5)
        {
            break;
        }
    }

    EXPECT_EQ(cnt, 10);
    fg->stop();

}
