gen_copy_impl_h = custom_target('gen_copy_impl_h',
                        input : [join_paths(meson.source_root(),'utils','blockbuilder','templates','blockname_impl.hpp.j2')],
                        output : ['copy_impl.hpp'],
                        command : [python3, join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_template.py'),'--input_file', '@INPUT@', '--output_file', '@OUTPUT@', '--yaml_file', join_paths(meson.source_root(),'blocklib/blocks/copy/copy.yml')])

gen_copy_impl_h = declare_dependency(sources : [gen_copy_impl_h], dependencies : [gen_copy_h])


gen_copy_cpu_h = custom_target('gen_copy_cpu_h',
                        input : [join_paths(meson.source_root(),'utils','blockbuilder','templates','blockname_domain.hpp.j2')],
                        output : ['copy_cpu.hpp'],
                        command : [python3, join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_template.py'),'--input_file', '@INPUT@', '--output_file', '@OUTPUT@', '--yaml_file', join_paths(meson.source_root(),'blocklib/blocks/copy/copy.yml')])

gen_copy_cpu_h = declare_dependency(sources : [gen_copy_cpu_h])

blocks_copy_deps = [gen_copy_h, gen_copy_impl_h, gen_copy_cpu_h]