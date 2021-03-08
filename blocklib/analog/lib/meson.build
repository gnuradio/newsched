sources = [
    'agc.cpp',
    'agc_blk.cpp'
]

incdir = include_directories(['../include/gnuradio/blocklib/analog','../include'])
newsched_blocklib_analog_lib = library('newsched-blocklib-analog', 
    sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep])

newsched_blocklib_analog_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_analog_lib)
