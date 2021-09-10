qtgui_inc_dir = join_paths('..','include','gnuradio','qtgui')

qt5_mod = import('qt5')
qt5widgets_dep = dependency('qt5', modules : 'Widgets')

qwt_dep = dependency('Qwt', method: 'cmake', cmake_module_path: join_paths(meson.source_root(), 'cmake','Modules') )

qtgui_sources += [
    'qtgui_util.cc',
    'displayform.cc',
    'spectrumUpdateEvents.cc',
    'timedisplayform.cc',
    'DisplayPlot.cc',
    'TimeDomainDisplayPlot.cc',
    'timecontrolpanel.cc'
    ]

moc_srcs = [
    join_paths(qtgui_inc_dir, 'displayform.h'),
    join_paths(qtgui_inc_dir, 'DisplayPlot.h'),
    join_paths(qtgui_inc_dir, 'form_menus.h'),
    join_paths(qtgui_inc_dir, 'timecontrolpanel.h'),
    join_paths(qtgui_inc_dir, 'timedisplayform.h'),
    join_paths(qtgui_inc_dir, 'TimeDomainDisplayPlot.h')
]
qtgui_moc_sources = qt5_mod.compile_moc(headers: moc_srcs)
qtgui_sources += qtgui_moc_sources


qtgui_deps += [newsched_runtime_dep, volk_dep, fmt_dep, pmt_dep, qt5widgets_dep, qwt_dep]

block_cpp_args = ['-DHAVE_CPU','-DQWT_DLL']

# if cuda_dep.found() and get_option('enable_cuda')
#     block_cpp_args += '-DHAVE_CUDA'

#     newsched_blocklib_qtgui_cu = library('newsched-blocklib-qtgui-cu', 
#         qtgui_cu_sources, 
#         include_directories : incdir, 
#         install : true, 
#         dependencies : [cuda_dep])

#     newsched_blocklib_qtgui_cu_dep = declare_dependency(include_directories : incdir,
#                         link_with : newsched_blocklib_qtgui_cu,
#                         dependencies : cuda_dep)

#     qtgui_deps += [newsched_blocklib_qtgui_cu_dep, cuda_dep]

# endif

incdir = include_directories(['../include/gnuradio/qtgui','../include'])
newsched_blocklib_qtgui_lib = library('newsched-blocklib-qtgui', 
    qtgui_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : qtgui_deps,
    cpp_args : block_cpp_args)

newsched_blocklib_qtgui_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_qtgui_lib,
                       dependencies : qtgui_deps)
