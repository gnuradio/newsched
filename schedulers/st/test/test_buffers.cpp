#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/head.hpp>
#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/null_sink.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/domain_adapter_shm.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/logging.hpp>
#include <gnuradio/schedulers/st/scheduler_st.hpp>
#include <gnuradio/vmcirc_buffer.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    auto logger = logging::get_logger("TEST_SCHED_ST", "default");

    // Basic test of the single threaded scheduler single instance
    if (1) {
        int samples = 1000000;
        float k = 100.0;
        std::vector<float> input_data(samples);
        std::vector<float> expected_output(samples);
        for (int i = 0; i < samples; i++) {
            input_data[i] = (float)i;
            expected_output[i] = input_data[i] * k;
        }
        auto src = blocks::vector_source_f::make(input_data, true);
        auto mult = blocks::multiply_const_ff::make(k);
        auto head = blocks::head::make(sizeof(float), samples);
        auto snk = blocks::vector_sink_f::make();

        auto fg(std::make_shared<flowgraph>());
        fg->connect(src, 0, mult, 0)->set_custom_buffer(VMCIRC_BUFFER_ARGS);
        fg->connect(mult, 0, head, 0)->set_custom_buffer(VMCIRC_BUFFER_ARGS);
        fg->connect(head, 0, snk, 0)->set_custom_buffer(VMCIRC_BUFFER_ARGS);

        std::shared_ptr<schedulers::scheduler_st> sched(new schedulers::scheduler_st());
        fg->set_scheduler(sched);

        fg->validate();
        fg->start();
        fg->wait();

        auto snk_data = snk->data();
        gr_log_info(
            logger, "valid output: {}, {}", snk_data == expected_output, snk_data.size());
    }
}
