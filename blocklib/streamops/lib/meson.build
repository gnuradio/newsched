sources = [
    'interleaved_short_to_complex.cpp',
]

incdir = include_directories(['../include/gnuradio/streamops','../include'])
newsched_blocklib_streamops_lib = library('newsched-blocklib-streamops', 
    sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep])

newsched_blocklib_streamops_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_streamops_lib)
