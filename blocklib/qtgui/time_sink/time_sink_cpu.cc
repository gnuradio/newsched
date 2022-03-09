/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "time_sink_cpu.h"
#include "time_sink_cpu_gen.h"
#include <volk/volk.h>

namespace gr {
namespace qtgui {

template <class T>
time_sink_cpu<T>::time_sink_cpu(const typename time_sink<T>::block_args& args)
    : INHERITED_CONSTRUCTORS(T),
      d_size(args.size),
      d_buffer_size(2 * args.size),
      d_samp_rate(args.samp_rate),
      d_name(args.name),
      d_nconnections(args.nconnections)
{
    if (sizeof(T) == sizeof(gr_complex)) {
        d_nconnections = 2 * args.nconnections;
    }
    // // setup PDU handling input port
    // message_port_register_in(pmtf::string("in"));
    // set_msg_handler(pmtf::string("in"), [this](pmtf::pmt msg) { this->handle_pdus(msg);
    // });

    // +1 for the PDU buffer
    for (unsigned int n = 0; n < d_nconnections + 1; n++) {
        d_buffers.emplace_back(d_buffer_size);
        d_Tbuffers.emplace_back(d_buffer_size);
    }

    // Set alignment properties for VOLK
    // const int alignment_multiple = volk_get_alignment() / sizeof(float);
    // set_alignment(std::max(1, alignment_multiple));

    d_tags = std::vector<std::vector<gr::tag_t>>(d_nconnections);

    initialize();

    d_main_gui->setNPoints(d_size); // setup GUI box with size
    // set_trigger_mode(TRIG_MODE_FREE, TRIG_SLOPE_POS, 0, 0, 0);

    // set_history(2);          // so we can look ahead for the trigger slope
    // declare_sample_delay(1); // delay the tags for a history of 2

    _reset();
}

template <>
void time_sink_cpu<float>::initialize()
{
    if (qApp != NULL) {
        d_qApplication = qApp;
    }
    else {
        d_qApplication = new QApplication(d_argc, &d_argv);
    }

    // If a style sheet is set in the prefs file, enable it here.
    // check_set_qss(d_qApplication);

    unsigned int numplots = (d_nconnections > 0) ? d_nconnections : 1;
    d_main_gui = new TimeDisplayForm(numplots, d_parent);
    d_main_gui->setNPoints(d_size);
    d_main_gui->setSampleRate(d_samp_rate);

    if (!d_name.empty())
        set_title(d_name);

    // initialize update time to 10 times a second
    set_update_time(0.1);
}

template <class T>
void time_sink_cpu<T>::initialize()
{
    if (qApp != NULL) {
        d_qApplication = qApp;
    }
    else {
        d_qApplication = new QApplication(d_argc, &d_argv);
    }

    // If a style sheet is set in the prefs file, enable it here.
    // check_set_qss(d_qApplication);

    unsigned int numplots = (d_nconnections > 0) ? d_nconnections : 2;
    d_main_gui = new TimeDisplayForm(numplots, d_parent);
    d_main_gui->setNPoints(d_size);
    d_main_gui->setSampleRate(d_samp_rate);

    if (!d_name.empty())
        set_title(d_name);

    // initialize update time to 10 times a second
    set_update_time(0.1);
}

template <class T>
void time_sink_cpu<T>::_reset()
{
    unsigned int n;
    if (d_trigger_delay) {
        for (n = 0; n < d_nconnections; n++) {
            // Move the tail of the buffers to the front. This section
            // represents data that might have to be plotted again if a
            // trigger occurs and we have a trigger delay set.  The tail
            // section is between (d_end-d_trigger_delay) and d_end.
            memmove(d_Tbuffers[n].data(),
                    &d_Tbuffers[n][d_end - d_trigger_delay],
                    d_trigger_delay * sizeof(float));

            // // Also move the offsets of any tags that occur in the tail
            // // section so they would be plotted again, too.
            // std::vector<gr::tag_t> tmp_tags;
            // for (size_t t = 0; t < d_tags[n].size(); t++) {
            //     if (d_tags[n][t].offset > (uint64_t)(d_size - d_trigger_delay)) {
            //         d_tags[n][t].offset =
            //             d_tags[n][t].offset - (d_size - d_trigger_delay);
            //         tmp_tags.push_back(d_tags[n][t]);
            //     }
            // }
            // d_tags[n] = tmp_tags;
        }
    }
    // Otherwise, just clear the local list of tags.
    else {
        for (n = 0; n < d_nconnections; n++) {
            d_tags[n].clear();
        }
    }

    // Reset the start and end indices.
    d_start = 0;
    d_end = d_size;

    // Reset the trigger. If in free running mode, ignore the
    // trigger delay and always set trigger to true.
    if (d_trigger_mode == TRIG_MODE_FREE) {
        d_index = 0;
        d_triggered = true;
    }
    else {
        d_index = d_trigger_delay;
        d_triggered = false;
    }
}

template <class T>
void time_sink_cpu<T>::_npoints_resize()
{
    int newsize = d_main_gui->getNPoints();
    set_nsamps(newsize);
}

template <class T>
void time_sink_cpu<T>::_adjust_tags(int adj)
{
    for (size_t n = 0; n < d_tags.size(); n++) {
        for (size_t t = 0; t < d_tags[n].size(); t++) {
            d_tags[n][t].set_offset(d_tags[n][t].offset() + adj);
        }
    }
}

template <class T>
void time_sink_cpu<T>::_gui_update_trigger()
{
    d_trigger_mode = d_main_gui->getTriggerMode();
    d_trigger_slope = d_main_gui->getTriggerSlope();
    d_trigger_level = d_main_gui->getTriggerLevel();
    d_trigger_channel = d_main_gui->getTriggerChannel();
    d_trigger_count = 0;

    float delayf = d_main_gui->getTriggerDelay();
    int delay = static_cast<int>(delayf * d_samp_rate);

    if (delay != d_trigger_delay) {
        // We restrict the delay to be within the window of time being
        // plotted.
        if ((delay < 0) || (delay >= d_size)) {
            this->d_logger->warn(
                        "Trigger delay ({}) outside of display range (0:{}).",
                        (delay / d_samp_rate),
                        ((d_size - 1) / d_samp_rate));
            delay = std::max(0, std::min(d_size - 1, delay));
            delayf = delay / d_samp_rate;
        }

        d_trigger_delay = delay;
        d_main_gui->setTriggerDelay(delayf);
        _reset();
    }

    // std::string tagkey = d_main_gui->getTriggerTagKey();
    // d_trigger_tag_key = pmtf::string(tagkey);
}

template <class T>
void time_sink_cpu<T>::_test_trigger_tags(int nitems)
{
    // int trigger_index;

    // uint64_t nr = nitems_read(d_trigger_channel);
    // std::vector<gr::tag_t> tags;
    // get_tags_in_range(tags, d_trigger_channel, nr, nr + nitems + 1, d_trigger_tag_key);
    // if (!tags.empty()) {
    //     trigger_index = tags[0].offset - nr;
    //     int start = d_index + trigger_index - d_trigger_delay - 1;
    //     if (start >= 0) {
    //         d_triggered = true;
    //         d_start = start;
    //         d_end = d_start + d_size;
    //         d_trigger_count = 0;
    //         _adjust_tags(-d_start);
    //     }
    // }
}


template <>
bool time_sink_cpu<gr_complex>::_test_trigger_slope(const gr_complex* in) const
{
    float x0, x1;
    if (d_trigger_channel % 2 == 0) {
        x0 = in[0].real();
        x1 = in[1].real();
    }
    else {
        x0 = in[0].imag();
        x1 = in[1].imag();
    }


    if (d_trigger_slope == TRIG_SLOPE_POS)
        return ((x0 <= d_trigger_level) && (x1 > d_trigger_level));
    else
        return ((x0 >= d_trigger_level) && (x1 < d_trigger_level));
}

template <class T>
bool time_sink_cpu<T>::_test_trigger_slope(const T* in) const
{
    float x0, x1;
    x0 = in[0];
    x1 = in[1];

    if (d_trigger_slope == TRIG_SLOPE_POS)
        return ((x0 <= d_trigger_level) && (x1 > d_trigger_level));
    else
        return ((x0 >= d_trigger_level) && (x1 < d_trigger_level));
}

template <>
void time_sink_cpu<gr_complex>::_test_trigger_norm(int nitems,
                                                   gr_vector_const_void_star inputs)
{
    int trigger_index;
    const gr_complex* in = (const gr_complex*)inputs[d_trigger_channel / 2];
    for (trigger_index = 0; trigger_index < nitems - 1; trigger_index++) {
        d_trigger_count++;

        // Test if trigger has occurred based on the input stream,
        // channel number, and slope direction
        if (_test_trigger_slope(&in[trigger_index])) {
            d_triggered = true;
            d_start = d_index + trigger_index - d_trigger_delay;
            d_end = d_start + d_size;
            d_trigger_count = 0;
            _adjust_tags(-d_start);
            break;
        }
    }

    // If using auto trigger mode, trigger periodically even
    // without a trigger event.
    if ((d_trigger_mode == TRIG_MODE_AUTO) && (d_trigger_count > d_size)) {
        d_triggered = true;
        d_trigger_count = 0;
    }
}


template <class T>
void time_sink_cpu<T>::_test_trigger_norm(int nitems, gr_vector_const_void_star inputs)
{
    int trigger_index;
    const float* in = (const float*)inputs[d_trigger_channel];
    for (trigger_index = 0; trigger_index < nitems; trigger_index++) {
        d_trigger_count++;

        // Test if trigger has occurred based on the input stream,
        // channel number, and slope direction
        if (_test_trigger_slope(&in[trigger_index])) {
            d_triggered = true;
            d_start = d_index + trigger_index - d_trigger_delay;
            d_end = d_start + d_size;
            d_trigger_count = 0;
            _adjust_tags(-d_start);
            break;
        }
    }

    // If using auto trigger mode, trigger periodically even
    // without a trigger event.
    if ((d_trigger_mode == TRIG_MODE_AUTO) && (d_trigger_count > d_size)) {
        d_triggered = true;
        d_trigger_count = 0;
    }
}


template <>
work_return_code_t
time_sink_cpu<float>::work(std::vector<block_work_input_sptr>& work_input,
                           std::vector<block_work_output_sptr>& work_output)
{
    auto noutput_items = work_input[0]->n_items; // need to check across all inputs

    unsigned int n = 0, idx = 0;
    const float* in;

    _npoints_resize();
    _gui_update_trigger();

    std::scoped_lock lock(d_setlock);

    int nfill = d_end - d_index;                 // how much room left in buffers
    int nitems = std::min(noutput_items, nfill); // num items we can put in buffers

    // If auto, normal, or tag trigger, look for the trigger
    if ((d_trigger_mode != TRIG_MODE_FREE) && !d_triggered) {
        // trigger off a tag key (first one found)
        if (d_trigger_mode == TRIG_MODE_TAG) {
            _test_trigger_tags(nitems);
        }
        // Normal or Auto trigger
        else {
            _test_trigger_norm(nitems, block_work_input::all_items(work_input));
        }
    }

    // Copy data into the buffers.
    for (n = 0; n < d_nconnections; n++) {
        in = work_input[idx]->items<float>();
        // memcpy(&d_Tbuffers[n][d_index], &in[1], nitems * sizeof(float));
        memcpy(&d_Tbuffers[n][d_index], &in[0], nitems * sizeof(float));
        // volk_32f_convert_64f(&d_buffers[n][d_index],
        //                     &in[1], nitems);

        // uint64_t nr = nitems_read(idx);
        // std::vector<gr::tag_t> tags;
        // get_tags_in_range(tags, idx, nr, nr + nitems + 1);
        // for (size_t t = 0; t < tags.size(); t++) {
        //     tags[t].offset = tags[t].offset - nr + (d_index - d_start - 1);
        // }
        // d_tags[idx].insert(d_tags[idx].end(), tags.begin(), tags.end());
        idx++;
    }
    d_index += nitems;

    // If we've have a trigger and a full d_size of items in the buffers, plot.
    if ((d_triggered) && (d_index == d_end)) {
        // Copy data to be plotted to start of buffers.
        for (n = 0; n < d_nconnections; n++) {
            // memmove(d_buffers[n], &d_buffers[n][d_start], d_size*sizeof(double));
            volk_32f_convert_64f(d_buffers[n].data(), &d_Tbuffers[n][d_start], d_size);
        }

        // Plot if we are able to update
        if (gr::high_res_timer_now() - d_last_time > d_update_time) {
            d_last_time = gr::high_res_timer_now();
            d_qApplication->postEvent(d_main_gui,
                                      new TimeUpdateEvent(d_buffers, d_size, d_tags));
        }

        // We've plotting, so reset the state
        _reset();
    }
    // If we've filled up the buffers but haven't triggered, reset.
    if (d_index == d_end) {
        _reset();
    }

    this->consume_each(nitems, work_input);
    return work_return_code_t::WORK_OK;
}


template <class T>
work_return_code_t
time_sink_cpu<T>::work(std::vector<block_work_input_sptr>& work_input,
                       std::vector<block_work_output_sptr>& work_output)
{
    auto noutput_items = work_input[0]->n_items; // need to check across all inputs

    unsigned int n = 0, idx = 0;
    const T* in;

    _npoints_resize();
    _gui_update_trigger();

    std::scoped_lock lock(d_setlock);

    int nfill = d_end - d_index;                 // how much room left in buffers
    int nitems = std::min(noutput_items, nfill); // num items we can put in buffers

    // If auto, normal, or tag trigger, look for the trigger
    if ((d_trigger_mode != TRIG_MODE_FREE) && !d_triggered) {
        // trigger off a tag key (first one found)
        if (d_trigger_mode == TRIG_MODE_TAG) {
            _test_trigger_tags(nitems);
        }
        // Normal or Auto trigger
        else {
            _test_trigger_norm(nitems, block_work_input::all_items(work_input));
        }
    }

    // Copy data into the buffers.
    for (n = 0; n < d_nconnections / 2; n++) {
        in = work_input[idx]->items<T>();
        // memcpy(&d_Tbuffers[n][d_index], &in[1], nitems * sizeof(float));
        memcpy(&d_Tbuffers[n][d_index], &in[0], nitems * sizeof(T));
        // volk_32f_convert_64f(&d_buffers[n][d_index],
        //                     &in[1], nitems);

        // uint64_t nr = nitems_read(idx);
        // std::vector<gr::tag_t> tags;
        // get_tags_in_range(tags, idx, nr, nr + nitems + 1);
        // for (size_t t = 0; t < tags.size(); t++) {
        //     tags[t].offset = tags[t].offset - nr + (d_index - d_start - 1);
        // }
        // d_tags[idx].insert(d_tags[idx].end(), tags.begin(), tags.end());
        idx++;
    }
    d_index += nitems;

    // If we've have a trigger and a full d_size of items in the buffers, plot.
    if ((d_triggered) && (d_index == d_end)) {
        // Copy data to be plotted to start of buffers.
        for (n = 0; n < d_nconnections / 2; n++) {
            volk_32fc_deinterleave_64f_x2(d_buffers[2 * n + 0].data(),
                                          d_buffers[2 * n + 1].data(),
                                          &d_Tbuffers[n][d_start],
                                          d_size);
        }

        // Plot if we are able to update
        if (gr::high_res_timer_now() - d_last_time > d_update_time) {
            d_last_time = gr::high_res_timer_now();
            d_qApplication->postEvent(d_main_gui,
                                      new TimeUpdateEvent(d_buffers, d_size, d_tags));
        }

        // We've plotting, so reset the state
        _reset();
    }

    // If we've filled up the buffers but haven't triggered, reset.
    if (d_index == d_end) {
        _reset();
    }

    this->consume_each(nitems, work_input);
    return work_return_code_t::WORK_OK;
}

template <class T>
void time_sink_cpu<T>::set_y_axis(double min, double max)
{
    d_main_gui->setYaxis(min, max);
}

template <class T>
void time_sink_cpu<T>::set_y_label(const std::string& label, const std::string& unit)
{
    d_main_gui->setYLabel(label, unit);
}

template <class T>
void time_sink_cpu<T>::set_update_time(double t)
{
    // convert update time to ticks
    gr::high_res_timer_type tps = gr::high_res_timer_tps();
    d_update_time = t * tps;
    d_main_gui->setUpdateTime(t);
    d_last_time = 0;
}

template <class T>
void time_sink_cpu<T>::set_title(const std::string& title)
{
    d_main_gui->setTitle(title.c_str());
}

template <class T>
void time_sink_cpu<T>::set_line_label(unsigned int which, const std::string& label)
{
    d_main_gui->setLineLabel(which, label.c_str());
}

template <class T>
void time_sink_cpu<T>::set_line_color(unsigned int which, const std::string& color)
{
    d_main_gui->setLineColor(which, color.c_str());
}

template <class T>
void time_sink_cpu<T>::set_line_width(unsigned int which, int width)
{
    d_main_gui->setLineWidth(which, width);
}

template <class T>
void time_sink_cpu<T>::set_line_style(unsigned int which, int style)
{
    d_main_gui->setLineStyle(which, (Qt::PenStyle)style);
}

template <class T>
void time_sink_cpu<T>::set_line_marker(unsigned int which, int marker)
{
    d_main_gui->setLineMarker(which, (QwtSymbol::Style)marker);
}

template <class T>
void time_sink_cpu<T>::set_line_alpha(unsigned int which, double alpha)
{
    d_main_gui->setMarkerAlpha(which, (int)(255.0 * alpha));
}

// void time_sink_cpu<T>::set_trigger_mode(trigger_mode mode,
//                                         trigger_slope slope,
//                                         float level,
//                                         float delay,
//                                         int channel,
//                                         const std::string& tag_key)
// {
//     gr::thread::scoped_lock lock(d_setlock);

//     d_trigger_mode = mode;
//     d_trigger_slope = slope;
//     d_trigger_level = level;
//     d_trigger_delay = static_cast<int>(delay * d_samp_rate);
//     d_trigger_channel = channel;
//     d_trigger_tag_key = pmtf::string(tag_key);
//     d_triggered = false;
//     d_trigger_count = 0;

//     if ((d_trigger_delay < 0) || (d_trigger_delay >= d_size)) {
//         GR_LOG_WARN(
//             d_logger,
//             "Trigger delay ({}) outside of display range (0:{}).",
//                 (d_trigger_delay / d_samp_rate) % ((d_size - 1) / d_samp_rate));
//         d_trigger_delay = std::max(0, std::min(d_size - 1, d_trigger_delay));
//         delay = d_trigger_delay / d_samp_rate;
//     }

//     d_main_gui->setTriggerMode(d_trigger_mode);
//     d_main_gui->setTriggerSlope(d_trigger_slope);
//     d_main_gui->setTriggerLevel(d_trigger_level);
//     d_main_gui->setTriggerDelay(delay);
//     d_main_gui->setTriggerChannel(d_trigger_channel);
//     d_main_gui->setTriggerTagKey(tag_key);

//     _reset();
// }

template <class T>
void time_sink_cpu<T>::set_size(int width, int height)
{
    d_main_gui->resize(QSize(width, height));
}

template <class T>
void time_sink_cpu<T>::set_nsamps(const int newsize)
{
    if (newsize != d_size) {
        std::scoped_lock lock(d_setlock);

        // Set new size and reset buffer index
        // (throws away any currently held data, but who cares?)
        d_size = newsize;
        d_buffer_size = 2 * d_size;

        // Resize buffers and replace data
        for (unsigned int n = 0; n < d_nconnections + 1; n++) {
            d_buffers[n].clear();
            d_buffers[n].resize(d_buffer_size);
            d_Tbuffers[n].clear();
            d_Tbuffers[n].resize(d_buffer_size);
        }

        // If delay was set beyond the new boundary, pull it back.
        if (d_trigger_delay >= d_size) {
            this->d_logger->warn(
                        "Trigger delay ({}) outside of display range "
                        "(0:{}). Moving to 50%% point.",
                        (d_trigger_delay / d_samp_rate),
                        ((d_size - 1) / d_samp_rate));
            d_trigger_delay = d_size - 1;
            d_main_gui->setTriggerDelay(d_trigger_delay / d_samp_rate);
        }

        d_main_gui->setNPoints(d_size);
        _reset();
    }
}

template <class T>
void time_sink_cpu<T>::set_samp_rate(const double samp_rate)
{
    std::scoped_lock lock(d_setlock);
    d_samp_rate = samp_rate;
    d_main_gui->setSampleRate(d_samp_rate);
}

} /* namespace qtgui */
} /* namespace gr */
