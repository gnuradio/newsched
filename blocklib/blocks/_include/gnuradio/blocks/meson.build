# copy_h = configure_file(copy: true, input : ['../../../copy/copy.hpp'], output : 'copy.hpp')
gen_copy_h = custom_target('jinja-output',
                        input : [join_paths(meson.source_root(),'utils','blockbuilder','templates','blockname.hpp.j2')],
                        output : ['copy.hpp'],
                        command : [python3, join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_template.py'),'--input_file', '@INPUT@', '--output_file', '@OUTPUT@', '--yaml_file', join_paths(meson.source_root(),'blocklib/blocks/copy/copy.yml')])

gen_copy_h = declare_dependency(sources : [gen_copy_h])

# gen_multiply_const_h = custom_target('gen-multiply-const-h',
#                         # input : [join_paths(meson.source_root(),'utils','blockbuilder','templates','blockname.hpp.j2')],
#                         input : [join_paths(meson.source_root(),'blocklib/blocks/multiply_const/blockname_templated.hpp.j2')],
#                         output : ['multiply_const.hpp'],
#                         command : [python3, join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_template.py'),'--input_file', '@INPUT@', '--output_file', '@OUTPUT@', '--yaml_file', join_paths(meson.source_root(),'blocklib/blocks/multiply_const/multiply_const.yml')])

# gen_multiply_const_h = declare_dependency(sources : [gen_multiply_const_h] )


blocks_headers = [
    'fanout.hpp',
    'head.hpp',
    'null_sink.hpp',
    'null_source.hpp',
    'vector_sink.hpp',
    'vector_source.hpp',
]

install_headers(blocks_headers, subdir : 'gnuradio/blocks')
