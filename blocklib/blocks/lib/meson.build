blocks_sources = [
    'annotator.cc',
    'copy.cc',
    'fanout.cc',
    'head.cc',
    'multiply_const.cc',
    'vector_sink.cc',
    'vector_source.cc'
]

incdir = include_directories(['../include/gnuradio/blocks','../include'])
newsched_blocklib_blocks_lib = library('newsched-blocklib-blocks', 
    blocks_sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep])

newsched_blocklib_blocks_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_blocks_lib)
