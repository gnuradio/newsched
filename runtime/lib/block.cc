#include <gnuradio/block.hh>
#include <gnuradio/scheduler.hh>
#include <pmtf/wrap.hpp>

#include <gnuradio/pyblock_detail.hh>
namespace gr {

block::block(const std::string& name,
          const std::string& module)
    : node(name), s_module(module), d_tag_propagation_policy(tag_propagation_policy_t::TPP_ALL_TO_ALL)
{
    // {# add message handler port for parameter updates#}
    _msg_param_update = message_port::make("param_update", port_direction_t::INPUT);
    _msg_param_update->register_callback(
        [this](pmtf::pmt msg) { this->handle_msg_param_update(msg); });
    add_port(_msg_param_update);
}

void block::set_pyblock_detail(std::shared_ptr<pyblock_detail> p)
{
    d_pyblock_detail = p;
}
std::shared_ptr<pyblock_detail> block::pb_detail() { return d_pyblock_detail; }
bool block::start()
{
    d_running = true;
    return true;
}

bool block::stop()
{
    d_running = false;
    return true;
}
bool block::done()
{
    d_running = false;
    return true;
}

tag_propagation_policy_t block::tag_propagation_policy()
{
    return d_tag_propagation_policy;
};

void block::set_tag_propagation_policy(tag_propagation_policy_t policy)
{
    d_tag_propagation_policy = policy;
};


void block::on_parameter_change(param_action_sptr action)
{
    gr_log_debug(
        _debug_logger, "block {}: on_parameter_change param_id: {}", id(), action->id());
    auto& param = d_parameters.get(action->id());
    param = action->pmt_value();
}

void block::on_parameter_query(param_action_sptr action)
{
    gr_log_debug(
        _debug_logger, "block {}: on_parameter_query param_id: {}", id(), action->id());
    auto param = d_parameters.get(action->id());
    action->set_pmt_value(param);
}

void block::consume_each(int num, std::vector<block_work_input_sptr>& work_input)
{
    for (auto& input : work_input) {
        input->consume(num);
    }
}

void block::produce_each(int num, std::vector<block_work_output_sptr>& work_output)
{
    for (auto& output : work_output) {
        output->produce(num);
    }
}

void block::set_output_multiple(int multiple)
{
    if (multiple < 1)
        throw std::invalid_argument("block::set_output_multiple");

    d_output_multiple_set = true;
    d_output_multiple = multiple;
}

void block::handle_msg_param_update(pmtf::pmt msg)
{
    // Update messages are a pmtf::map with the name of
    // the param as the "id" field, and the pmt::wrap
    // that holds the update as the "value" field

    auto id = pmtf::get_string(pmtf::get_map(msg)["id"]).data();
    auto value = pmtf::get_map(msg)["value"];

    request_parameter_change(get_param_id(id), value, false);
}

void block::request_parameter_change(int param_id, pmtf::pmt new_value, bool block)
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

        if (block) {
            // block until confirmation that parameter has been set
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk);
        }
    }
    // else go ahead and update parameter value
    else {
        on_parameter_change(param_action::make(param_id, new_value, 0));
    }
}

pmtf::pmt block::request_parameter_query(int param_id)
{
    // call back to the scheduler if ptr is not null
    if (p_scheduler && d_running) {
        std::condition_variable cv;
        std::mutex m;
        pmtf::pmt newval;
        auto lam = [&](param_action_sptr a) {
            std::unique_lock<std::mutex> lk(m);
            newval = a->pmt_value();
            cv.notify_one();
        };

        auto msg =
            std::make_shared<param_query_action>(id(), param_action::make(param_id), lam);
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

void block::notify_scheduler()
{
    if (p_scheduler) {
        this->p_scheduler->push_message(
            std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_ALL));
    }
}

void block::notify_scheduler_input()
{
    if (p_scheduler) {
        this->p_scheduler->push_message(
            std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_INPUT));
    }
}

void block::notify_scheduler_output()
{
    if (p_scheduler) {
        this->p_scheduler->push_message(
            std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_OUTPUT));
    }
}


} // namespace gr
