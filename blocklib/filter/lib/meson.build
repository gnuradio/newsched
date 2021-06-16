sources = [
    'moving_averager.cc',
    'fir_filter.cc',
    'mmse_fir_interpolator_ff.cc',
    'mmse_interpolator_ff.cc'
]

filter_sources += sources
filter_deps += [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep]

block_cpp_args = ['-DHAVE_CPU']
if cuda_dep.found() and get_option('enable_cuda')
    block_cpp_args += '-DHAVE_CUDA'

    newsched_blocklib_filter_cu = library('newsched-blocklib-filter-cu', 
        filter_cu_sources, 
        include_directories : incdir, 
        install : true, 
        dependencies : [cuda_dep])

    newsched_blocklib_filter_cu_dep = declare_dependency(include_directories : incdir,
                        link_with : newsched_blocklib_filter_cu,
                        dependencies : cuda_dep)

    filter_deps += [newsched_blocklib_filter_cu_dep, cuda_dep]

endif

incdir = include_directories(['../include/gnuradio/filter','../include'])
newsched_blocklib_filter_lib = library('newsched-blocklib-filter', 
    filter_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : filter_deps,
    cpp_args : block_cpp_args)

newsched_blocklib_filter_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_filter_lib,
                       dependencies : filter_deps)
