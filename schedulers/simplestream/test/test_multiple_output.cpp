#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/dummy.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    auto src = blocks::vector_source_f::make(
        std::vector<float>{ 1.0, 2.0, 3.0, 4.0, 5.0 }, true);
    auto throttle = blocks::throttle::make(sizeof(float), 100);
    auto dummy = blocks::dummy<float>::make(7.0,13.0);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();
    // blocks::vector_sink_f snk();

    // mult->on_parameter_change(std::vector<param_action_base>{
    // param_action<float>(blocks::multiply_const_ff::params::id_k, 12.0, 0) });

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src->base(), 0, throttle->base(), 0);
    fg->connect(throttle->base(), 0, dummy->base(), 0);
    fg->connect(dummy->base(), 0, snk1->base(), 0);
    fg->connect(dummy->base(), 1, snk2->base(), 0);
    // fg->connect(mult->base(), 0, snk->base(), 0);


    std::shared_ptr<schedulers::scheduler_simplestream> sched(
        new schedulers::scheduler_simplestream());
    fg->set_scheduler(sched->base());

    fg->validate();

    // what happens when k is set here???

    fg->start();

    auto start = std::chrono::steady_clock::now();

    float a = 1.0;
    float b = 100.0;
    while (true) {
        std::cout << "query a is " << dummy->a() << std::endl;
        std::cout << "query b is " << dummy->b() << std::endl;

        dummy->set_a(a);
        dummy->set_b(b);

        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(200))
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        a += 1.0;
        b += 1.0;

        for (const auto& d : snk1->data())
            std::cout << d << ' ';
        std::cout << std::endl;
    
        for (const auto& d : snk2->data())
            std::cout << d << ' ';
        std::cout << std::endl;


    }

    fg->stop();
    fg->wait();

    // DOMAIN??

    // now look at the data
}