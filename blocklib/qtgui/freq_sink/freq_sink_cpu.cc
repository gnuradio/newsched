/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "freq_sink_cpu.hh"
#include "freq_sink_cpu_gen.hh"
#include <volk/volk.h>
#include <pmtf/string.hpp>

namespace gr {
namespace qtgui {

template <class T>
freq_sink_cpu<T>::freq_sink_cpu(const typename freq_sink<T>::block_args& args)
    : sync_block("freq_sink"),
      freq_sink<T>(args),
      d_fftsize(args.fftsize),
      //   d_fft_shift(fftsize),
      d_fftavg(1.0),
      d_wintype((fft::window::win_type)(args.wintype)),
      d_window_normalize(args.wintype & (1 << 15)),
      d_center_freq(args.fc),
      d_bandwidth(args.bw),
      d_name(args.name),
      d_nconnections(args.nconnections),
      d_port(pmtf::string("freq")),
      d_port_bw(pmtf::string("bw"))
{
    d_fft = std::make_unique<fft::fft_complex_fwd>(d_fftsize);
    d_fbuf.resize(d_fftsize);

    // save the last "connection" for the PDU memory
    for (int i = 0; i < d_nconnections + 1; i++) {
        d_residbufs.emplace_back(d_fftsize);
        d_magbufs.emplace_back(d_fftsize);
    }

    d_pdu_magbuf = d_magbufs[d_magbufs.size() - 1].data();

    buildwindow();

    initialize();

    set_trigger_mode(TRIG_MODE_FREE, 0, 0);
}

template <class T>
freq_sink_cpu<T>::~freq_sink_cpu<T>()
{
    if (!d_main_gui->isClosed())
        d_main_gui->close();
}

template <class T>
work_return_code_t freq_sink_cpu<T>::work(std::vector<block_work_input_sptr>& work_input,
                                          std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0]->items<T>();
    auto noutput_items = work_input[0]->n_items;

    // Update the FFT size from the application
    bool updated = false;
    updated |= fftresize();
    updated |= windowreset();
    if (updated) {
        this->consume_each(0, work_input);
        return work_return_code_t::WORK_OK;
    }

    check_clicked();
    _gui_update_trigger();

    std::scoped_lock lock(d_setlock);
    for (d_index = 0; d_index < noutput_items; d_index += d_fftsize) {

        if ((gr::high_res_timer_now() - d_last_time) > d_update_time) {

            // Trigger off tag, if active
            if ((d_trigger_mode == TRIG_MODE_TAG) && !d_triggered) {
                _test_trigger_tags(d_index, d_fftsize);
                if (d_triggered) {
                    // If not enough from tag position, early exit
                    if ((d_index + d_fftsize) >= noutput_items) {
                        this->consume_each(d_index, work_input);
                        return work_return_code_t::WORK_OK;
                    }
                }
            }

            // Perform FFT and shift operations into d_magbufs
            for (int n = 0; n < d_nconnections; n++) {
                in = work_input[n]->items<T>();
                memcpy(
                    d_residbufs[n].data(), &in[d_index], sizeof(gr_complex) * d_fftsize);

                fft(d_fbuf.data(), d_residbufs[n].data(), d_fftsize);
                for (int x = 0; x < d_fftsize; x++) {
                    d_magbufs[n][x] = (double)((1.0 - d_fftavg) * d_magbufs[n][x] +
                                               (d_fftavg)*d_fbuf[x]);
                }
                // volk_32f_convert_64f(d_magbufs[n], d_fbuf, d_fftsize);
            }

            // Test trigger off signal power in d_magbufs
            if ((d_trigger_mode == TRIG_MODE_NORM) ||
                (d_trigger_mode == TRIG_MODE_AUTO)) {
                _test_trigger_norm(d_fftsize, d_magbufs);
            }

            // If a trigger (FREE always triggers), plot and reset state
            if (d_triggered) {
                d_last_time = gr::high_res_timer_now();
                d_qApplication->postEvent(d_main_gui,
                                          new FreqUpdateEvent(d_magbufs, d_fftsize));
                _reset();
            }
        }
    }

    this->consume_each(noutput_items, work_input);
    return work_return_code_t::WORK_OK;
}

template <class T>
void freq_sink_cpu<T>::initialize()
{
    if (qApp != NULL) {
        d_qApplication = qApp;
    } else {
#if QT_VERSION >= 0x040500 && QT_VERSION < 0x050000
        std::string style = prefs::singleton()->get_string("qtgui", "style", "raster");
        QApplication::setGraphicsSystem(QString(style.c_str()));
#endif
        d_qApplication = new QApplication(d_argc, &d_argv);
    }

    // If a style sheet is set in the prefs file, enable it here.
    check_set_qss(d_qApplication);

    int numplots = (d_nconnections > 0) ? d_nconnections : 1;
    d_main_gui = new FreqDisplayForm(numplots, d_parent);
    set_fft_window(d_wintype);
    set_fft_size(d_fftsize);
    set_frequency_range(d_center_freq, d_bandwidth);

    if (!d_name.empty())
        set_title(d_name);

    this->set_output_multiple(d_fftsize);

    // initialize update time to 10 times a second
    set_update_time(0.1);
}

template <class T>
void freq_sink_cpu<T>::exec_()
{
    d_qApplication->exec();
}

template <class T>
QWidget* freq_sink_cpu<T>::qwidget()
{
    return d_main_gui;
}

template <class T>
void freq_sink_cpu<T>::set_fft_size(const int fftsize)
{
    if ((fftsize > 16) && (fftsize < 16384))
        d_main_gui->setFFTSize(fftsize);
    else
        throw std::runtime_error("freq_sink: FFT size must be > 16 and < 16384.");
}

template <class T>
int freq_sink_cpu<T>::fft_size() const
{
    return d_fftsize;
}

template <class T>
void freq_sink_cpu<T>::set_fft_average(const float fftavg)
{
    d_main_gui->setFFTAverage(fftavg);
}

template <class T>
float freq_sink_cpu<T>::fft_average() const
{
    return d_fftavg;
}

template <class T>
void freq_sink_cpu<T>::set_fft_window(const fft::window::win_type win)
{
    d_main_gui->setFFTWindowType(win);
}

template <class T>
fft::window::win_type freq_sink_cpu<T>::fft_window() const
{
    return d_wintype;
}

template <class T>
void freq_sink_cpu<T>::set_fft_window_normalized(const bool enable)
{
    d_window_normalize = enable;
    buildwindow();
}

template <class T>
void freq_sink_cpu<T>::set_frequency_range(const double centerfreq,
                                           const double bandwidth)
{
    d_center_freq = centerfreq;
    d_bandwidth = bandwidth;
    d_main_gui->setFrequencyRange(d_center_freq, d_bandwidth);
}

template <class T>
void freq_sink_cpu<T>::set_y_axis(double min, double max)
{
    d_main_gui->setYaxis(min, max);
}

template <class T>
void freq_sink_cpu<T>::set_y_label(const std::string& label, const std::string& unit)
{
    d_main_gui->setYLabel(label, unit);
}

template <class T>
void freq_sink_cpu<T>::set_update_time(double t)
{
    // convert update time to ticks
    gr::high_res_timer_type tps = gr::high_res_timer_tps();
    d_update_time = t * tps;
    d_main_gui->setUpdateTime(t);
    d_last_time = 0;
}

template <class T>
void freq_sink_cpu<T>::set_title(const std::string& title)
{
    d_main_gui->setTitle(title.c_str());
}

template <class T>
void freq_sink_cpu<T>::set_line_label(unsigned int which, const std::string& label)
{
    d_main_gui->setLineLabel(which, label.c_str());
}

template <class T>
void freq_sink_cpu<T>::set_line_color(unsigned int which, const std::string& color)
{
    d_main_gui->setLineColor(which, color.c_str());
}

template <class T>
void freq_sink_cpu<T>::set_line_width(unsigned int which, int width)
{
    d_main_gui->setLineWidth(which, width);
}

template <class T>
void freq_sink_cpu<T>::set_line_style(unsigned int which, int style)
{
    d_main_gui->setLineStyle(which, (Qt::PenStyle)style);
}

template <class T>
void freq_sink_cpu<T>::set_line_marker(unsigned int which, int marker)
{
    d_main_gui->setLineMarker(which, (QwtSymbol::Style)marker);
}

template <class T>
void freq_sink_cpu<T>::set_line_alpha(unsigned int which, double alpha)
{
    d_main_gui->setMarkerAlpha(which, (int)(255.0 * alpha));
}

template <class T>
void freq_sink_cpu<T>::set_size(int width, int height)
{
    d_main_gui->resize(QSize(width, height));
}

template <class T>
void freq_sink_cpu<T>::set_trigger_mode(trigger_mode mode,
                                        float level,
                                        int channel,
                                        const std::string& tag_key)
{
    std::scoped_lock lock(d_setlock);

    d_trigger_mode = mode;
    d_trigger_level = level;
    d_trigger_channel = channel;
    d_trigger_tag_key = pmtf::string(tag_key);
    d_triggered = false;
    d_trigger_count = 0;

    d_main_gui->setTriggerMode(d_trigger_mode);
    d_main_gui->setTriggerLevel(d_trigger_level);
    d_main_gui->setTriggerChannel(d_trigger_channel);
    d_main_gui->setTriggerTagKey(tag_key);

    _reset();
}

template <class T>
std::string freq_sink_cpu<T>::title()
{
    return d_main_gui->title().toStdString();
}

template <class T>
std::string freq_sink_cpu<T>::line_label(unsigned int which)
{
    return d_main_gui->lineLabel(which).toStdString();
}

template <class T>
std::string freq_sink_cpu<T>::line_color(unsigned int which)
{
    return d_main_gui->lineColor(which).toStdString();
}

template <class T>
int freq_sink_cpu<T>::line_width(unsigned int which)
{
    return d_main_gui->lineWidth(which);
}

template <class T>
int freq_sink_cpu<T>::line_style(unsigned int which)
{
    return d_main_gui->lineStyle(which);
}

template <class T>
int freq_sink_cpu<T>::line_marker(unsigned int which)
{
    return d_main_gui->lineMarker(which);
}

template <class T>
double freq_sink_cpu<T>::line_alpha(unsigned int which)
{
    return (double)(d_main_gui->markerAlpha(which)) / 255.0;
}

template <class T>
void freq_sink_cpu<T>::enable_menu(bool en)
{
    d_main_gui->enableMenu(en);
}

template <class T>
void freq_sink_cpu<T>::enable_grid(bool en)
{
    d_main_gui->setGrid(en);
}

template <class T>
void freq_sink_cpu<T>::enable_autoscale(bool en)
{
    d_main_gui->autoScale(en);
}

template <class T>
void freq_sink_cpu<T>::enable_axis_labels(bool en)
{
    d_main_gui->setAxisLabels(en);
}

template <class T>
void freq_sink_cpu<T>::enable_control_panel(bool en)
{
    if (en)
        d_main_gui->setupControlPanel();
    else
        d_main_gui->teardownControlPanel();
}

template <class T>
void freq_sink_cpu<T>::enable_max_hold(bool en)
{
    d_main_gui->notifyMaxHold(en);
}

template <class T>
void freq_sink_cpu<T>::enable_min_hold(bool en)
{
    d_main_gui->notifyMinHold(en);
}

template <class T>
void freq_sink_cpu<T>::clear_max_hold()
{
    d_main_gui->clearMaxHold();
}

template <class T>
void freq_sink_cpu<T>::clear_min_hold()
{
    d_main_gui->clearMinHold();
}

template <class T>
void freq_sink_cpu<T>::disable_legend()
{
    d_main_gui->disableLegend();
}

template <class T>
void freq_sink_cpu<T>::reset()
{
    std::scoped_lock lock(d_setlock);
    _reset();
}

template <class T>
void freq_sink_cpu<T>::_reset()
{
    d_trigger_count = 0;

    // Reset the trigger.
    if (d_trigger_mode == TRIG_MODE_FREE) {
        d_triggered = true;
    } else {
        d_triggered = false;
    }
}

template <>
void freq_sink_cpu<gr_complex>::fft(float* data_out, const gr_complex* data_in, int size)
{
    if (!d_window.empty()) {
        volk_32fc_32f_multiply_32fc(d_fft->get_inbuf(), data_in, &d_window.front(), size);
    } else {
        memcpy(d_fft->get_inbuf(), data_in, sizeof(gr_complex) * size);
    }

    d_fft->execute(); // compute the fft

    volk_32fc_s32f_x2_power_spectral_density_32f(
        data_out, d_fft->get_outbuf(), size, 1.0, size);

    // d_fft_shift.shift(data_out, size);
}

template <>
void freq_sink_cpu<float>::fft(float* data_out, const float* data_in, int size)
{
    // float to complex conversion
    gr_complex* dst = d_fft->get_inbuf();
    for (int i = 0; i < size; i++)
        dst[i] = data_in[i];

    if (!d_window.empty()) {
        volk_32fc_32f_multiply_32fc(d_fft->get_inbuf(), dst, &d_window.front(), size);
    }

    d_fft->execute(); // compute the fft
    volk_32fc_s32f_x2_power_spectral_density_32f(
        data_out, d_fft->get_outbuf(), size, 1.0, size);

    // d_fft_shift.shift(data_out, size);
}

template <class T>
void freq_sink_cpu<T>::fft(float* data_out, const T* data_in, int size)
{
    throw std::runtime_error("Should not get in here");
}

template <class T>
bool freq_sink_cpu<T>::windowreset()
{
    std::scoped_lock lock(d_setlock);

    fft::window::win_type newwintype;
    newwintype = d_main_gui->getFFTWindowType();
    if (d_wintype != newwintype) {
        d_wintype = newwintype;
        buildwindow();
        return true;
    }
    return false;
}

template <class T>
void freq_sink_cpu<T>::buildwindow()
{
    d_window.clear();
    if (d_wintype != fft::window::WIN_NONE) {
        d_window = fft::window::build(d_wintype, d_fftsize, 6.76, d_window_normalize);
    }
}

template <class T>
bool freq_sink_cpu<T>::fftresize()
{
    std::scoped_lock lock(d_setlock);

    int newfftsize = d_main_gui->getFFTSize();
    d_fftavg = d_main_gui->getFFTAverage();

    if (newfftsize != d_fftsize) {
        // Resize residbuf and replace data
        // +1 to handle PDU buffers
        for (int i = 0; i < d_nconnections + 1; i++) {
            d_residbufs[i].clear();
            d_residbufs[i].resize(newfftsize);
            d_magbufs[i].clear();
            d_magbufs[i].resize(newfftsize);
        }

        // Update the pointer to the newly allocated memory
        d_pdu_magbuf = d_magbufs[d_nconnections].data();

        // Set new fft size and reset buffer index
        // (throws away any currently held data, but who cares?)
        d_fftsize = newfftsize;
        d_index = 0;

        // Reset window to reflect new size
        buildwindow();

        // Reset FFTW plan for new size
        d_fft = std::make_unique<fft::fft_complex_fwd>(d_fftsize);

        d_fbuf.clear();
        d_fbuf.resize(d_fftsize);

        // d_fft_shift.resize(d_fftsize);

        d_last_time = 0;

        this->set_output_multiple(d_fftsize);

        return true;
    }
    return false;
}

template <class T>
void freq_sink_cpu<T>::check_clicked()
{
    // if (d_main_gui->checkClicked()) {
    //     double freq = d_main_gui->getClickedFreq();
    //     message_port_pub(d_port, pmt::cons(d_port, pmt::from_double(freq)));
    // }
}

// template <class T>
// void freq_sink_cpu<T>::handle_set_freq(pmtf::wrap msg)
// {
//     if (pmt::is_pair(msg)) {
//         pmtf::wrap x = pmt::cdr(msg);
//         if (pmt::is_real(x)) {
//             d_center_freq = pmt::to_double(x);
//             d_qApplication->postEvent(d_main_gui,
//                                       new SetFreqEvent(d_center_freq, d_bandwidth));
//         }
//     }
// }

// template <class T>
// void freq_sink_cpu<T>::handle_set_bw(pmtf::wrap msg)
// {
//     if (pmt::is_pair(msg)) {
//         pmtf::wrap x = pmt::cdr(msg);
//         if (pmt::is_real(x)) {
//             d_bandwidth = pmt::to_double(x);
//             d_qApplication->postEvent(d_main_gui,
//                                       new SetFreqEvent(d_center_freq, d_bandwidth));
//         }
//     }
// }

template <class T>
void freq_sink_cpu<T>::_gui_update_trigger()
{
    trigger_mode new_trigger_mode = d_main_gui->getTriggerMode();
    d_trigger_level = d_main_gui->getTriggerLevel();
    d_trigger_channel = d_main_gui->getTriggerChannel();

    std::string tagkey = d_main_gui->getTriggerTagKey();
    d_trigger_tag_key = pmtf::string(tagkey);

    if (new_trigger_mode != d_trigger_mode) {
        d_trigger_mode = new_trigger_mode;
        _reset();
    }
}

template <class T>
void freq_sink_cpu<T>::_test_trigger_tags(int start, int nitems)
{
    // uint64_t nr = nitems_read(d_trigger_channel);
    // std::vector<gr::tag_t> tags;
    // get_tags_in_range(
    //     tags, d_trigger_channel, nr + start, nr + start + nitems, d_trigger_tag_key);
    // if (!tags.empty()) {
    //     d_triggered = true;
    //     d_index = tags[0].offset - nr;
    //     d_trigger_count = 0;
    // }
}

template <class T>
void freq_sink_cpu<T>::_test_trigger_norm(int nitems,
                                          std::vector<volk::vector<double>> inputs)
{
    const double* in = (const double*)inputs[d_trigger_channel].data();
    for (int i = 0; i < nitems; i++) {
        d_trigger_count++;

        // Test if trigger has occurred based on the FFT magnitude and
        // channel number. Test if any value is > the level (in dBx).
        if (in[i] > d_trigger_level) {
            d_triggered = true;
            d_trigger_count = 0;
            break;
        }
    }

    // If using auto trigger mode, trigger periodically even
    // without a trigger event.
    if ((d_trigger_mode == TRIG_MODE_AUTO) && (d_trigger_count > d_fftsize)) {
        d_triggered = true;
        d_trigger_count = 0;
    }
}

// template <class T>
// void freq_sink_cpu<T>::handle_pdus(pmtf::wrap msg)
// {
//     size_t len;
//     pmtf::wrap dict, samples;

//     // Test to make sure this is either a PDU or a uniform vector of
//     // samples. Get the samples PMT and the dictionary if it's a PDU.
//     // If not, we throw an error and exit.
//     if (pmt::is_pair(msg)) {
//         dict = pmt::car(msg);
//         samples = pmt::cdr(msg);
//     } else if (pmt::is_uniform_vector(msg)) {
//         samples = msg;
//     } else {
//         throw std::runtime_error("time_sink_c: message must be either "
//                                  "a PDU or a uniform vector of samples.");
//     }

//     len = pmt::length(samples);

//     const gr_complex* in;
//     if (pmt::is_c32vector(samples)) {
//         in = (const gr_complex*)pmt::c32vector_elements(samples, len);
//     } else {
//         throw std::runtime_error("freq_sink_c: unknown data type "
//                                  "of samples; must be complex.");
//     }

//     // Plot if we're past the last update time
//     if (gr::high_res_timer_now() - d_last_time > d_update_time) {
//         d_last_time = gr::high_res_timer_now();

//         // Update the FFT size from the application
//         fftresize();
//         windowreset();
//         check_clicked();

//         int winoverlap = 4;
//         int fftoverlap = d_fftsize / winoverlap;
//         float num = static_cast<float>(winoverlap * len) / static_cast<float>(d_fftsize);
//         int nffts = static_cast<int>(ceilf(num));

//         // Clear this as we will be accumulating in the for loop over nffts
//         memset(d_pdu_magbuf, 0, sizeof(double) * d_fftsize);

//         size_t min = 0;
//         size_t max = std::min(d_fftsize, static_cast<int>(len));
//         for (int n = 0; n < nffts; n++) {
//             // Clear in case (max-min) < d_fftsize
//             std::fill(std::begin(d_residbufs[d_nconnections]),
//                       std::end(d_residbufs[d_nconnections]),
//                       0x00);

//             // Copy in as much of the input samples as we can
//             memcpy(d_residbufs[d_nconnections].data(),
//                    &in[min],
//                    sizeof(gr_complex) * (max - min));

//             // Apply the window and FFT; copy data into the PDU
//             // magnitude buffer.
//             fft(d_fbuf.data(), d_residbufs[d_nconnections].data(), d_fftsize);
//             for (int x = 0; x < d_fftsize; x++) {
//                 d_pdu_magbuf[x] += (double)d_fbuf[x];
//             }

//             // Increment our indices; set max up to the number of
//             // samples in the input PDU.
//             min += fftoverlap;
//             max = std::min(max + fftoverlap, len);
//         }

//         // Perform the averaging
//         for (int x = 0; x < d_fftsize; x++) {
//             d_pdu_magbuf[x] /= static_cast<double>(nffts);
//         }

//         // update gui per-pdu
//         d_qApplication->postEvent(d_main_gui, new FreqUpdateEvent(d_magbufs, d_fftsize));
//     }
// }

template <class T>
void freq_sink_cpu<T>::set_plot_pos_half(bool half)
{
    d_main_gui->setPlotPosHalf(half);
}

} /* namespace qtgui */
} /* namespace gr */
