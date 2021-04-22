blocks_sources = [
    'vector_sink.cpp',
    'vector_source.cpp',
    blocks_copy_files,
    blocks_annotator_files,
    blocks_head_files,
    blocks_multiply_const_files
]

blocks_deps = [
    blocks_copy_deps,
    blocks_annotator_deps,
    blocks_head_deps,
    blocks_multiply_const_deps
]

incdir = include_directories(['../include/gnuradio/blocks','../include'])
newsched_blocklib_blocks_lib = library('newsched-blocklib-blocks', 
    blocks_sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep, blocks_deps])

newsched_blocklib_blocks_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_blocks_lib,
                       dependencies : blocks_deps)
