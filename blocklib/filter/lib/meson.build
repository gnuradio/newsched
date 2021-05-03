sources = [
    'dc_blocker.cpp',
    'fir_filter.cpp',
    'mmse_fir_interpolator_ff.cpp',
    'mmse_interpolator_ff.cpp'
]

incdir = include_directories(['../include/gnuradio/filter','../include'])
newsched_blocklib_filter_lib = library('newsched-blocklib-filter', 
    sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep])

newsched_blocklib_filter_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_filter_lib)
