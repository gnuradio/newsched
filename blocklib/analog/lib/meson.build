analog_deps += [newsched_runtime_dep, volk_dep, fmt_dep, pmt_dep]

analog_sources += 'kernel/agc.cc'
block_cpp_args = ['-DHAVE_CPU']
# if cuda_dep.found() and get_option('enable_cuda')
#     block_cpp_args += '-DHAVE_CUDA'

#     newsched_blocklib_analog_cu = library('newsched-blocklib-analog-cu', 
#         analog_cu_sources, 
#         include_directories : incdir, 
#         install : true, 
#         dependencies : [cuda_dep])

#     newsched_blocklib_analog_cu_dep = declare_dependency(include_directories : incdir,
#                         link_with : newsched_blocklib_analog_cu,
#                         dependencies : cuda_dep)

#     analog_deps += [newsched_blocklib_analog_cu_dep, cuda_dep]

# endif

incdir = include_directories(['../include/gnuradio/analog','../include'])
newsched_blocklib_analog_lib = library('newsched-blocklib-analog', 
    analog_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : analog_deps,
    cpp_args : block_cpp_args)

newsched_blocklib_analog_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_analog_lib,
                       dependencies : analog_deps)
