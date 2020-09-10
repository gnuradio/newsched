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

using namespace gr;

int main(int argc, char* argv[])
{
    auto logger = logging::get_logger("TEST_SCHED_ST", "debug");

    // Basic test of the single threaded scheduler single instance
    if (0) {
        int samples = 10000;
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
        fg->connect(src, 0, mult, 0);
        fg->connect(mult, 0, head, 0);
        fg->connect(head, 0, snk, 0);

        std::shared_ptr<schedulers::scheduler_st> sched(
            new schedulers::scheduler_st());
        fg->set_scheduler(sched);

        fg->validate();
        fg->start();
        fg->wait();

        auto snk_data = snk->data();
        gr_log_debug(
            logger, "valid output: {}, {}", snk_data == expected_output, snk_data.size());
    }

    // Split the blocks across two instances using domain adapters
    if (1) {
        int samples = 1000000;
        float k = 100.0;
        std::vector<float> input_data(samples);
        std::vector<float> expected_output(samples);
        for (int i = 0; i < samples; i++) {
            input_data[i] = (float)i;
            expected_output[i] = input_data[i] * k * k;
        }
        auto src = blocks::vector_source_f::make(input_data, true);
        auto mult1 = blocks::multiply_const_ff::make(k);
        auto mult2 = blocks::multiply_const_ff::make(k);
        auto head = blocks::head::make(sizeof(float), samples);
        auto snk = blocks::vector_sink_f::make();

        auto fg(std::make_shared<flowgraph>());
        fg->connect(src, 0, mult1, 0);
        fg->connect(mult1, 0, mult2, 0);
        fg->connect(mult2, 0, head, 0);
        fg->connect(head, 0, snk, 0);

        auto sched1 = std::make_shared<schedulers::scheduler_st>("sched1");
        auto sched2 = std::make_shared<schedulers::scheduler_st>("sched2");

        fg->add_scheduler(sched1);
        fg->add_scheduler(sched2);

        auto da_conf = domain_adapter_shm_conf::make(buffer_preference_t::UPSTREAM);

        domain_conf_vec dconf{ domain_conf(sched1, { src, mult1 }, da_conf),
                               domain_conf(sched2, { mult2, head, snk }, da_conf) };

        fg->partition(dconf);


        // fg->validate();
        fg->start();
        fg->wait();

        auto snk_data = snk->data();
        gr_log_info(
            logger, "valid output: {}, {}", snk_data == expected_output, snk_data.size());
    }

    // Asynchronous parameter queries and changes
    if (0) {
        int samples = 100000;
        float k = 100.0;
        std::vector<float> input_data(samples);
        for (int i = 0; i < samples; i++) {
            input_data[i] = (float)i;
        }
        auto src = blocks::vector_source_f::make(input_data, true);
        auto mult = blocks::multiply_const_ff::make(k);
        auto throttle = blocks::throttle::make(sizeof(float), 10000);
        // auto snk = blocks::null_sink::make(sizeof(float));
        auto snk = blocks::vector_sink_f::make();

        auto fg(std::make_shared<flowgraph>());
        fg->connect(src, 0, mult, 0);
        fg->connect(mult, 0, throttle, 0);
        fg->connect(throttle, 0, snk, 0);

        auto sched1 =
            std::make_shared<schedulers::scheduler_st>("sched1", 8192);
        fg->add_scheduler(sched1);

        fg->validate();

        // fg->validate();
        fg->start();
        auto start = std::chrono::steady_clock::now();

        while (true) {
            std::cout << "... "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count()
                      << std::endl;
            auto kk = mult->k();
            std::cout << "query k is: " << kk << std::endl;

            mult->set_k(k);

            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5))
                break;

            std::this_thread::sleep_for(std::chrono::milliseconds(300));

            k += 1.0;
        }

        fg->stop();

        std::cout << snk->data().size();
    }
}
