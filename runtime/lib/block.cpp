
#include <gnuradio/block.hpp>
#include <gnuradio/scheduler.hpp>

namespace gr {
void block::request_parameter_change(int param_id, pmtf::pmt_sptr new_value)
{
    // call back to the scheduler if ptr is not null
    if (p_scheduler && d_running) {
        std::condition_variable cv;
        std::mutex m;
        auto lam = [&](param_action_sptr a) {
            std::unique_lock<std::mutex> lk(m);
            cv.notify_one();
        };

        p_scheduler->push_message(std::make_shared<param_change_action>(
            id(), param_action::make(param_id, new_value, 0), lam));

        // block until confirmation that parameter has been set
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk);
    }
    // else go ahead and update parameter value
    else {
        on_parameter_change(param_action::make(param_id, new_value, 0));
    }
}

pmtf::pmt_sptr block::request_parameter_query(int param_id)
{
    // call back to the scheduler if ptr is not null
    if (p_scheduler && d_running) {
        std::condition_variable cv;
        std::mutex m;
        pmtf::pmt_sptr newval;
        auto lam = [&](param_action_sptr a) {
            std::unique_lock<std::mutex> lk(m);
            newval = a->pmt_value();
            cv.notify_one();
        };

        auto msg = std::make_shared<param_query_action>(
            id(), param_action::make(param_id), lam);
        p_scheduler->push_message(msg);

        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk);
        return newval;
    }
    // else go ahead and return parameter value
    else {
        auto action = param_action::make(param_id);
        on_parameter_query(action);
        return action->pmt_value();
    }
}
} // namespace gr