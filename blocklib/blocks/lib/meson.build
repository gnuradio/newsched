blocks_sources = [
    'annotator.cpp',
    'copy.cpp',
    'fanout.cpp',
    'head.cpp',
    'multiply_const.cpp',
    'vector_sink.cpp',
    'vector_source.cpp'
]

incdir = include_directories(['../include/gnuradio/blocklib/blocks','../include'])
newsched_blocklib_blocks_lib = library('newsched-blocklib-blocks', 
    blocks_sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep])

newsched_blocklib_blocks_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_blocks_lib)
