qtgui_inc_dir = join_paths('..','include','gnuradio','qtgui')


qtgui_sources += [
    'qtgui_util.cc',
    'displayform.cc',
    'spectrumUpdateEvents.cc',
    'timedisplayform.cc',
    'freqdisplayform.cc',
    'freqcontrolpanel.cc',
    'DisplayPlot.cc',
    'TimeDomainDisplayPlot.cc',
    'FrequencyDisplayPlot.cc',
    'timecontrolpanel.cc'
    ]

moc_srcs = [
    join_paths(qtgui_inc_dir, 'displayform.h'),
    join_paths(qtgui_inc_dir, 'DisplayPlot.h'),
    join_paths(qtgui_inc_dir, 'form_menus.h'),
    join_paths(qtgui_inc_dir, 'timecontrolpanel.h'),
    join_paths(qtgui_inc_dir, 'freqcontrolpanel.h'),
    join_paths(qtgui_inc_dir, 'timedisplayform.h'),
    join_paths(qtgui_inc_dir, 'freqdisplayform.h'),
    join_paths(qtgui_inc_dir, 'TimeDomainDisplayPlot.h'),
    join_paths(qtgui_inc_dir, 'FrequencyDisplayPlot.h')
]
qtgui_moc_sources = qt5_mod.compile_moc(headers: moc_srcs)
qtgui_sources += qtgui_moc_sources


qtgui_deps += [gnuradio_gr_dep, gnuradio_blocklib_filter_dep, volk_dep, fmt_dep, pmtf_dep, qt5widgets_dep, qwt_dep]

block_cpp_args = ['-DHAVE_CPU','-DQWT_DLL']

# if cuda_dep.found() and get_option('enable_cuda')
#     block_cpp_args += '-DHAVE_CUDA'

#     gnuradio_blocklib_qtgui_cu = library('gnuradio-blocklib-qtgui-cu', 
#         qtgui_cu_sources, 
#         include_directories : incdir, 
#         install : true, 
#         dependencies : [cuda_dep])

#     gnuradio_blocklib_qtgui_cu_dep = declare_dependency(include_directories : incdir,
#                         link_with : gnuradio_blocklib_qtgui_cu,
#                         dependencies : cuda_dep)

#     qtgui_deps += [gnuradio_blocklib_qtgui_cu_dep, cuda_dep]

# endif

incdir = include_directories(['../include/gnuradio/qtgui','../include'])
gnuradio_blocklib_qtgui_lib = library('gnuradio-blocklib-qtgui', 
    qtgui_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : qtgui_deps,
    cpp_args : block_cpp_args)

gnuradio_blocklib_qtgui_dep = declare_dependency(include_directories : incdir,
					   link_with : gnuradio_blocklib_qtgui_lib,
                       dependencies : qtgui_deps)
