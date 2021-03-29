blocks_sources = [
    '../annotator/annotator.cpp',
    '../copy/copy_cpu.cpp',
    '../fanout/fanout.cpp',
    '../head/head.cpp',
    '../multiply_const/multiply_const_cpu.cpp',
    '../vector_sink/vector_sink.cpp',
    '../vector_source/vector_source.cpp'
]

deps = [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep]
deps += blocks_copy_deps
deps += blocks_multiply_const_deps 

incdir = include_directories(['../_include/gnuradio/blocks','../_include'])
newsched_blocklib_blocks_lib = library('newsched-blocklib-blocks', 
    blocks_sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : deps)

newsched_blocklib_blocks_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_blocks_lib)
