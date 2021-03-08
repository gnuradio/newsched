sources = [
    'fast_atan2f.cc'
]

incdir = include_directories(['../include/gnuradio/blocklib/math','../include'])
newsched_blocklib_math_lib = library('newsched-blocklib-math', 
    sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep])

newsched_blocklib_math_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_math_lib)
