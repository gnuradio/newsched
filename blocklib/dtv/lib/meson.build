sources = [
    'atsc_fpll.cpp',
]

incdir = include_directories(['../include/gnuradio/blocklib/dtv','../include'])
newsched_blocklib_dtv_lib = library('newsched-blocklib-dtv', 
    sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep,
                    newsched_blocklib_math_dep, newsched_blocklib_filter_dep])

newsched_blocklib_dtv_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_dtv_lib)
