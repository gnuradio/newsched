#include "throttle_cpu.hh"

namespace gr {
namespace blocks {

throttle::sptr throttle::make_cpu(const block_args& args)
{
    return std::make_shared<throttle_cpu>(args);
}

void throttle_cpu::set_sample_rate(double rate)
{
    // changing the sample rate performs a reset of state params
    d_start = std::chrono::steady_clock::now();
    d_total_samples = 0;
    d_sample_rate = rate;
    d_sample_period = std::chrono::duration<double>(1 / rate);
}

bool throttle_cpu::start()
{
    d_start = std::chrono::steady_clock::now();
    d_total_samples = 0;
    return block::start();
}

work_return_code_t throttle_cpu::work(std::vector<block_work_input>& work_input,
                                      std::vector<block_work_output>& work_output)
{
    // check for updated rx_rate tag
    // if (!d_ignore_tags) {
    // uint64_t abs_N = work_input[0].n_items_read;
    // std::vector<tag_t> all_tags;
    // get_tags_in_range(all_tags, 0, abs_N, abs_N + noutput_items);
    // for (const auto& tag : all_tags) {
    //     if (pmt::eq(tag.key, throttle_rx_rate_pmt)) {
    //         double new_rate = pmt::to_double(tag.value);
    //         set_sample_rate(new_rate);
    //     }
    // }
    // }

    // copy all samples output[i] <= input[i]
    auto in = work_input[0].items<uint8_t>();
    auto out = work_output[0].items<uint8_t>();

    auto noutput_items = work_output[0].n_items;

    d_total_samples += noutput_items;

    auto now = std::chrono::steady_clock::now();
    auto expected_time = d_start + d_sample_period * d_total_samples;

    int n = noutput_items;
    if (expected_time > now) {
        auto limit_duration =
            std::chrono::duration<double>(std::numeric_limits<long>::max());
        // if (expected_time - now > limit_duration) {
        //     GR_LOG_ALERT(d_logger,
        //                  "WARNING: Throttle sleep time overflow! You "
        //                  "are probably using a very low sample rate.");
        // }

        // We should no longer block inside of a work function since it may not have its
        // own thread
        // This should only be done in its own thread
        std::this_thread::sleep_until(expected_time);

        // need a more intelligent solution to inform the scheduler
        // to return at a certain time
        n = 0;
        d_total_samples -= noutput_items;
    }

    // TODO: blocks like throttle shouldn't need to do a memcpy, but this would have to be
    // fixed in the buffering model and a special port type
    std::memcpy(out, in, n * d_itemsize);
    work_output[0].n_produced = n;
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr