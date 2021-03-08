#include <gnuradio/blocklib/analog/agc.hpp>

namespace gr {
namespace analog {
namespace kernel {

template <>
gr_complex agc<gr_complex>::scale(gr_complex input)
{
    gr_complex output = input * _gain;

    _gain += _rate * (_reference - std::sqrt(output.real() * output.real() +
                                             output.imag() * output.imag()));
    if (_max_gain > 0.0 && _gain > _max_gain) {
        _gain = _max_gain;
    }
    return output;
}

template <>
float agc<float>::scale(float input)
{
    float output = input * _gain;
    _gain += (_reference - fabsf(output)) * _rate;
    if (_max_gain > 0.0 && _gain > _max_gain)
        _gain = _max_gain;
    return output;
}

template class agc<float>;
template class agc<gr_complex>;

} // namespace kernel
} // namespace analog
} // namespace gr
