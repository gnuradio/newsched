#include <iostream>

#include <gnuradio/blocklib/blocks/multiply_const_blk.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    std::shared_ptr<blocks::multiply_const_ff> mult(
        new blocks::multiply_const_ff(17.0)); // create a block that multiplies by 17
    std::shared_ptr<blocks::vector_source_f> src(
        new blocks::vector_source_f(std::vector<float>({ 1.0, 2.0, 3.0, 4.0, 5.0 })));
    std::shared_ptr<blocks::vector_sink_f> snk(new blocks::vector_sink_f());
    // blocks::vector_sink_f snk();

    mult->on_parameter_change(std::vector<param_change_base>{
        param_change<float>(blocks::multiply_const_ff::params::k, 12.0, 0) });

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src->base(), 0, mult->base(), 0);
    fg->connect(mult->base(), 0, snk->base(), 0);
    

    std::shared_ptr<schedulers::scheduler_simplestream> sched(new schedulers::scheduler_simplestream());
    fg->set_scheduler(sched->base());

    fg->validate(); 
    fg->start();
    fg->wait();

    // DOMAIN??


    // sched.start();
    // sched.wait();

    for (const auto& d : snk->data())
        std::cout << d << ' ';
    std::cout << std::endl;

    // now look at the data
}