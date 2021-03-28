gen_copy_impl_h = custom_target('gen_copy_impl_h',
                        input : [join_paths(meson.source_root(),'blocklib/blocks/copy/copy_impl.hpp.j2')],
                        output : ['copy_impl.hpp'],
                        command : [python3, join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_template.py'),'--input_file', '@INPUT@', '--output_file', '@OUTPUT@', '--yaml_file', join_paths(meson.source_root(),'blocklib/blocks/copy/copy.yml')])

gen_copy_impl_h = declare_dependency(sources : [gen_copy_impl_h])


gen_copy_cpu_h = custom_target('gen_copy_cpu_h',
                        input : [join_paths(meson.source_root(),'blocklib/blocks/copy/copy_cpu.hpp.j2')],
                        output : ['copy_cpu.hpp'],
                        command : [python3, join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_template.py'),'--input_file', '@INPUT@', '--output_file', '@OUTPUT@', '--yaml_file', join_paths(meson.source_root(),'blocklib/blocks/copy/copy.yml')])

gen_copy_cpu_h = declare_dependency(sources : [gen_copy_cpu_h])

blocks_sources = [
    '../annotator/annotator.cpp',
    '../copy/copy_cpu.cpp',
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
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep, gen_copy_impl_h, gen_copy_cpu_h])

newsched_blocklib_blocks_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_blocks_lib)
