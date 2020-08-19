
#include <gnuradio/block.hpp>

#include <gnuradio/scheduler.hpp>

namespace gr {
// block::~block() {}
block::block(const std::string& name) : node(name) {}


template <class T>
void block::request_parameter_change(int param_id, T new_value)
{
    // call back to the scheduler if ptr is not null
    if (p_scheduler && d_running) {
        std::condition_variable cv;
        std::mutex m;
        auto lam = [&](param_action_sptr a) {
            std::unique_lock<std::mutex> lk(m);
            cv.notify_one();
        };
        p_scheduler->request_parameter_change(
            alias(), param_action<T>::make(param_id, new_value, 0), lam);

        // block until confirmation that parameter has been set
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk);
    }
    // else go ahead and update parameter value
    else {
        on_parameter_change(param_action<T>::make(param_id, new_value, 0));
    }
}

template <class T>
T block::request_parameter_query(int param_id)
{
    // call back to the scheduler if ptr is not null
    if (p_scheduler && d_running) {
        std::condition_variable cv;
        std::mutex m;
        T newval;
        auto lam = [&](param_action_sptr a) {
            std::unique_lock<std::mutex> lk(m);
            newval = std::static_pointer_cast<param_action<T>>(a)->new_value();
            cv.notify_one();
        };

        p_scheduler->request_parameter_query(
            alias(), param_action<T>::make(param_id), lam);

        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk);
        return newval;
    }
    // else go ahead and return parameter value
    else {
        auto action = param_action<T>::make(param_id);
        on_parameter_query(action);
        return action->new_value();
    }
}

// Acceptable parameter query types (has to be a better way to do this)
template float block::request_parameter_query<float>(int);
template double block::request_parameter_query<double>(int);
template gr_complex block::request_parameter_query<gr_complex>(int);
template int8_t block::request_parameter_query<int8_t>(int);
template int16_t block::request_parameter_query<int16_t>(int);
template int32_t block::request_parameter_query<int32_t>(int);
template int64_t block::request_parameter_query<int64_t>(int);
template uint8_t block::request_parameter_query<uint8_t>(int);
template uint16_t block::request_parameter_query<uint16_t>(int);
template uint32_t block::request_parameter_query<uint32_t>(int);
template uint64_t block::request_parameter_query<uint64_t>(int);

template std::vector<float> block::request_parameter_query<std::vector<float>>(int);
template std::vector<double> block::request_parameter_query<std::vector<double>>(int);
template std::vector<gr_complex>
block::request_parameter_query<std::vector<gr_complex>>(int);
template std::vector<int8_t> block::request_parameter_query<std::vector<int8_t>>(int);
template std::vector<int16_t> block::request_parameter_query<std::vector<int16_t>>(int);
template std::vector<int32_t> block::request_parameter_query<std::vector<int32_t>>(int);
template std::vector<int64_t> block::request_parameter_query<std::vector<int64_t>>(int);
template std::vector<uint8_t> block::request_parameter_query<std::vector<uint8_t>>(int);
template std::vector<uint16_t> block::request_parameter_query<std::vector<uint16_t>>(int);
template std::vector<uint32_t> block::request_parameter_query<std::vector<uint32_t>>(int);
template std::vector<uint64_t> block::request_parameter_query<std::vector<uint64_t>>(int);

template void block::request_parameter_change<float>(int, float);
template void block::request_parameter_change<double>(int, double);
template void block::request_parameter_change<gr_complex>(int, gr_complex);
template void block::request_parameter_change<int8_t>(int, int8_t);
template void block::request_parameter_change<int16_t>(int, int16_t);
template void block::request_parameter_change<int32_t>(int, int32_t);
template void block::request_parameter_change<int64_t>(int, int64_t);
template void block::request_parameter_change<uint8_t>(int, uint8_t);
template void block::request_parameter_change<uint16_t>(int, uint16_t);
template void block::request_parameter_change<uint32_t>(int, uint32_t);
template void block::request_parameter_change<uint64_t>(int, uint64_t);

template void block::request_parameter_change<std::vector<float>>(int,
                                                                  std::vector<float>);
template void block::request_parameter_change<std::vector<double>>(int,
                                                                   std::vector<double>);
template void
block::request_parameter_change<std::vector<gr_complex>>(int, std::vector<gr_complex>);
template void block::request_parameter_change<std::vector<int8_t>>(int,
                                                                   std::vector<int8_t>);
template void block::request_parameter_change<std::vector<int16_t>>(int,
                                                                    std::vector<int16_t>);
template void block::request_parameter_change<std::vector<int32_t>>(int,
                                                                    std::vector<int32_t>);
template void block::request_parameter_change<std::vector<int64_t>>(int,
                                                                    std::vector<int64_t>);
template void block::request_parameter_change<std::vector<uint8_t>>(int,
                                                                    std::vector<uint8_t>);
template void
block::request_parameter_change<std::vector<uint16_t>>(int, std::vector<uint16_t>);
template void
block::request_parameter_change<std::vector<uint32_t>>(int, std::vector<uint32_t>);
template void
block::request_parameter_change<std::vector<uint64_t>>(int, std::vector<uint64_t>);


} // namespace gr