#pragma once

#include <gnuradio/qtgui/time_sink.hh>

#include <gnuradio/high_res_timer.hh>
#include <gnuradio/qtgui/timedisplayform.h>

namespace gr {
namespace qtgui {

template <class T>
class time_sink_cpu : public time_sink<T>
{
public:
    time_sink_cpu(const typename time_sink<T>::block_args& args);

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

    virtual void exec_() { d_qApplication->exec(); };
    virtual QWidget* qwidget() { return d_main_gui; };
    // virtual void set_y_axis(double, double


    // ){};

protected:
    TimeDisplayForm* d_main_gui = nullptr;
    QApplication* d_qApplication;
};


} // namespace qtgui
} // namespace gr
