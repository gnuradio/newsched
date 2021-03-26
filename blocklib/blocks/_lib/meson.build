blocks_sources = [
    '../annotator/annotator.cpp',
    '../copy/copy.cpp',
    '../fanout/fanout.cpp',
    '../head/head.cpp',
    '../multiply_const/multiply_const.cpp',
    '../vector_sink/vector_sink.cpp',
    '../vector_source/vector_source.cpp'
]

incdir = include_directories(['../_include/gnuradio/blocks','../_include'])
newsched_blocklib_blocks_lib = library('newsched-blocklib-blocks', 
    blocks_sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep])

newsched_blocklib_blocks_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_blocks_lib)
