blocks_sources = [
    'annotator.cpp',
    'fanout.cpp',
    'head.cpp',
    'multiply_const.cpp',
    'vector_sink.cpp',
    'vector_source.cpp',
    blocks_copy_files
]

incdir = include_directories(['../include/gnuradio/blocks','../include'])
newsched_blocklib_blocks_lib = library('newsched-blocklib-blocks', 
    blocks_sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep, blocks_copy_deps])

newsched_blocklib_blocks_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_blocks_lib,
                       dependencies : blocks_copy_deps)
