#include <gnuradio/block.h>
#include <gnuradio/pyblock_detail.h>
#include <gnuradio/scheduler.h>
#include <gnuradio/scheduler_message.h>
#include <nlohmann/json.hpp>
#include <pmtf/wrap.hpp>
#include <atomic>
#include <chrono>
#include <thread>


#include <gnuradio/buffer_pdu.h>

namespace gr {

block::block(const std::string& name, const std::string& module)
    : node(name),
      s_module(module),
      d_tag_propagation_policy(tag_propagation_policy_t::TPP_ALL_TO_ALL)
{
    // {# add message handler port for parameter updates#}
    _msg_param_update = message_port::make("param_update", port_direction_t::INPUT);
    _msg_param_update->register_callback(
        [this](pmtf::pmt msg) { this->handle_msg_param_update(msg); });
    add_port(_msg_param_update);

    _msg_system = message_port::make("system", port_direction_t::INPUT);
    _msg_system->register_callback(
        [this](pmtf::pmt msg) { this->handle_msg_system(msg); });
    add_port(_msg_system);

    _msg_work = message_port::make("pdus_in", port_direction_t::INPUT);
    _msg_work->register_callback([this](pmtf::pmt msg) { this->handle_msg_work(msg); });
    add_port(_msg_work);
    _msg_work_out = message_port::make("pdus_out", port_direction_t::OUTPUT);
    add_port(_msg_work_out);
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
    d_debug_logger->debug(
        "block {}: on_parameter_change param_id: {}", id(), action->id());
    auto param = d_parameters.get(action->id());
    *param = action->pmt_value();
}

void block::on_parameter_query(param_action_sptr action)
{
    d_debug_logger->debug(
        "block {}: on_parameter_query param_id: {}", id(), action->id());
    auto param = d_parameters.get(action->id());
    action->set_pmt_value(*param);
}

void block::consume_each(size_t num, std::vector<block_work_input_sptr>& work_input)
{
    for (auto& input : work_input) {
        input->consume(num);
    }
}

void block::produce_each(size_t num, std::vector<block_work_output_sptr>& work_output)
{
    for (auto& output : work_output) {
        output->produce(num);
    }
}

void block::set_output_multiple(size_t multiple)
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

    auto id = pmtf::string(pmtf::map(msg)["id"]).data();
    auto value = pmtf::map(msg)["value"];

    request_parameter_change(get_param_id(id), value, false);
}

void block::handle_msg_work(pmtf::pmt msg)
{

    // only considering 1 input and 1 output for now
    // FIXME: need checks elsewhere to enforce this

    // prepare the input buffer
    // Interpret the data based on the port that it represents
    auto input_port = this->get_port(0, port_type_t::STREAM, port_direction_t::INPUT);
    auto output_port = this->get_port(0, port_type_t::STREAM, port_direction_t::OUTPUT);

    // The PDU must match the input port
    auto meta = pmtf::map(msg)["meta"];
    auto data = pmtf::map(msg)["data"];

    // data should be a vector of some sort
    uint8_t* input_items = nullptr;
    size_t num_input_items = 0;
    size_t input_itemsize = input_port->itemsize();  // size in bytes of an item
    size_t input_datasize = input_port->data_size(); // size in bytes of an individual element 
    auto vlen = input_itemsize / input_datasize ;
    switch (input_port->data_type()) {
    case param_type_t::FLOAT: {
        auto vec = pmtf::vector<float>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    case param_type_t::DOUBLE: {
        auto vec = pmtf::vector<double>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    case param_type_t::CFLOAT: {
        auto vec = pmtf::vector<gr_complex>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    case param_type_t::CDOUBLE: {
        auto vec = pmtf::vector<gr_complexd>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    case param_type_t::INT8: {
        auto vec = pmtf::vector<int8_t>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    case param_type_t::INT16: {
        auto vec = pmtf::vector<int16_t>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    case param_type_t::INT32: {
        auto vec = pmtf::vector<int32_t>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    case param_type_t::INT64: {
        auto vec = pmtf::vector<int64_t>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    case param_type_t::UINT8: {
        auto vec = pmtf::vector<uint8_t>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    case param_type_t::UINT16: {
        auto vec = pmtf::vector<uint16_t>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    case param_type_t::UINT32: {
        auto vec = pmtf::vector<uint32_t>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    case param_type_t::UINT64: {
        auto vec = pmtf::vector<uint64_t>(data);
        input_items = reinterpret_cast<uint8_t*>(vec.data());
        num_input_items = vec.size() / vlen;
    } break;
    default:
        break;
    }

    
    auto br = buffer_pdu_reader::make(num_input_items, input_itemsize, input_items, msg);


    // data should be a vector of some sort
    uint8_t* output_items = nullptr;
    size_t output_itemsize = output_port->itemsize();
    size_t output_datasize = output_port->data_size();
    auto output_vlen = output_itemsize / output_datasize;

    size_t num_output_items = static_cast<size_t>(num_input_items * this->relative_rate());
    pmtf::pmt output_vec;

    switch (output_port->data_type()) {
    case param_type_t::FLOAT: {
        auto vec = pmtf::vector<float>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::DOUBLE: {
        auto vec = pmtf::vector<double>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::CFLOAT: {
        auto vec = pmtf::vector<gr_complex>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::CDOUBLE: {
        auto vec = pmtf::vector<gr_complexd>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::INT8: {
        auto vec = pmtf::vector<int8_t>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::INT16: {
        auto vec = pmtf::vector<int16_t>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::INT32: {
        auto vec = pmtf::vector<int32_t>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::INT64: {
        auto vec = pmtf::vector<int64_t>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::UINT8: {
        auto vec = pmtf::vector<uint8_t>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::UINT16: {
        auto vec = pmtf::vector<uint16_t>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::UINT32: {
        auto vec = pmtf::vector<uint32_t>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::UINT64: {
        auto vec = pmtf::vector<uint64_t>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    case param_type_t::UNTYPED: {
        // FIXME: there is no way untyped ports will work with this ...
        auto vec = pmtf::vector<uint8_t>(num_output_items);
        output_items = reinterpret_cast<uint8_t*>(vec.data());
        output_vec = vec;
    } break;
    default:
        break;
    }

    auto bw = buffer_pdu::make(num_output_items, output_itemsize, output_items, output_vec);

    std::vector<block_work_input_sptr> work_input;
    std::vector<block_work_output_sptr> work_output;
    work_input.push_back(std::make_shared<block_work_input>(num_input_items, br));
    work_output.push_back(std::make_shared<block_work_output>(num_output_items, bw));

    auto code = work(work_input, work_output);

    if (code == work_return_code_t::WORK_OK)
    {
        // // validate the n_produced
        // if (work_output[0]->n_produced < num_output_items)
        // {
        //     switch (output_port->data_type()) {
        //     case param_type_t::FLOAT: {
        //         pmtf::vector<float>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::DOUBLE: {
        //         pmtf::vector<double>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::CFLOAT: {
        //         pmtf::vector<gr_complex>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::CDOUBLE: {
        //         pmtf::vector<gr_complexd>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::INT8: {
        //         pmtf::vector<int8_t>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::INT16: {
        //         pmtf::vector<int16_t>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::INT32: {
        //         pmtf::vector<int32_t>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::INT64: {
        //         pmtf::vector<int64_t>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::UINT8: {
        //         pmtf::vector<uint8_t>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::UINT16: {
        //         pmtf::vector<uint16_t>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::UINT32: {
        //         pmtf::vector<uint32_t>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::UINT64: {
        //         pmtf::vector<uint64_t>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     case param_type_t::UNTYPED: {
        //         // FIXME: there is no way untyped ports will work with this ...
        //         pmtf::vector<uint8_t>(output_vec).resize(work_output[0]->n_produced);
        //     } break;
        //     default:
        //         break;
        //     }
        // }

        auto pdu = pmtf::map({ { "data", output_vec }, { "meta", meta } });
        _msg_work_out->post(pdu);
    }
    else
    {
        // TODO: have a better call here
        throw std::runtime_error("Generic PDU handling on work port unable to handle this work call");
    }
}

void block::handle_msg_system(pmtf::pmt msg)
{
    auto str_msg = pmtf::get_as<std::string>(msg);
    if (str_msg == "done") {
        d_finished = true;
        p_scheduler->push_message(
            std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_ALL, id()));
    }
}

void block::request_parameter_change(int param_id, pmtf::pmt new_value, bool block)
{
    if (rpc_client() && !rpc_name().empty()) {
        rpc_client()->block_parameter_change(
            rpc_name(), get_param_str(param_id), new_value.to_base64());
    }
    else if (p_scheduler && d_running) {
        std::condition_variable cv;
        std::mutex m;
        bool ready{ false };
        auto lam = [&](param_action_sptr a) {
            {
                std::unique_lock<std::mutex> lk(m);
                ready = true;
            }
            cv.notify_all();
        };

        p_scheduler->push_message(std::make_shared<param_change_action>(
            id(), param_action::make(param_id, new_value, 0), lam));

        if (block) {
            // block until confirmation that parameter has been set
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&ready]() { return ready == true; });
        }
    }
    // else go ahead and update parameter value
    else {
        on_parameter_change(param_action::make(param_id, new_value, 0));
    }
}

pmtf::pmt block::request_parameter_query(int param_id)
{

    if (rpc_client() && !rpc_name().empty()) {
        auto encoded_str =
            rpc_client()->block_parameter_query(rpc_name(), get_param_str(param_id));
        return pmtf::pmt::from_base64(encoded_str);
    }
    // call back to the scheduler if ptr is not null
    else if (p_scheduler && d_running) {
        std::condition_variable cv;
        std::mutex m;
        pmtf::pmt newval;
        bool ready{ false };
        auto lam = [&](param_action_sptr a) {
            {
                std::unique_lock<std::mutex> lk(m);
                newval = a->pmt_value();
                ready = true;
            }
            cv.notify_all();
        };

        auto msg =
            std::make_shared<param_query_action>(id(), param_action::make(param_id), lam);
        p_scheduler->push_message(msg);

        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&ready]() { return ready == true; });
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

std::string block::to_json()
{
    // Example string describing this block
    // {"module": "blocks", "id": "copy", "properties": {"itemsize": 8}}

    nlohmann::json json_obj;

    json_obj["module"] = s_module;
    json_obj["id"] = name() + suffix();
    json_obj["format"] = "b64";
    for (auto [key, val] : d_parameters.param_map) {
        auto encoded_str = val->to_base64();
        json_obj["parameters"][key] = encoded_str;
    }

    return json_obj.dump();
}

void block::from_json(const std::string& json_str)
{
    using json = nlohmann::json;
    auto json_obj = json::parse(json_str);
    for (auto& [key, value] : json_obj["parameters"].items()) {
        // deserialize from the b64 string
        auto p = pmtf::pmt::from_base64(value.get<std::string>());
        auto block_pmt = d_parameters.get(key);
        *block_pmt = p;
    }
}

// This should go into pmt
pmtf::pmt block::deserialize_param_to_pmt(const std::string& encoded_str)
{
    return pmtf::pmt::from_base64(encoded_str);
}


void block::come_back_later(size_t count_ms)
{
    if (!p_scheduler) {
        return;
    }
    // Launch a thread to come back and try again some time later

    std::atomic<bool> d_sleeping = true;
    std::thread t([this, count_ms]() {
        d_debug_logger->debug("Setting timer to notify scheduler in {} ms", count_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(count_ms));
        std::atomic<bool> d_sleeping = false;
        p_scheduler->push_message(
            std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_INPUT));
    });
    t.detach();
}

} // namespace gr
