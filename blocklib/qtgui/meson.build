if (get_option('enable_gr_qtgui'))

    QTGUI_DEPS = qt5_mod.found() and qt5widgets_dep.found() and qwt_dep.found()
    if (QTGUI_DEPS)

    qtgui_sources = []
    qtgui_cu_sources = []
    qtgui_pybind_sources = []
    qtgui_pybind_names = []
    qtgui_deps = []

    subdir('include/gnuradio/qtgui')

    # Individual block subdirectories
    subdir('time_sink')
    subdir('freq_sink')


    subdir('lib')

    if (get_option('enable_python'))
        subdir('python/qtgui')
    endif

    if (get_option('enable_testing'))
        subdir('test')
    endif
    endif
endif