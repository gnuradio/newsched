
yml_file = join_paths(meson.source_root(),'blocklib/blocks/multiply_const/multiply_const.yml')

# gen_multiply_const_impl_h = custom_target('gen_multiply_const_impl_h',
#                         # input : [join_paths(meson.source_root(),'utils','blockbuilder','templates','blockname_impl.hpp.j2')],
#                         input : [join_paths(meson.source_root(),'blocklib/blocks/multiply_const/blockname_templated_impl.hpp.j2'),
#                                  yml_file],
#                         output : ['multiply_const_impl.hpp'],
#                         command : [python3, join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_template.py'),'--input_file', '@INPUT0@', '--output_file', '@OUTPUT@', '--yaml_file', '@INPUT1@'])

# gen_multiply_const_impl_h = declare_dependency(sources : [gen_multiply_const_impl_h], dependencies : [gen_multiply_const_h])

gen_multiply_const_cpu_h = custom_target('gen_multiply_const_cpu_h',
                        # input : [join_paths(meson.source_root(),'utils','blockbuilder','templates','blockname_domain.hpp.j2')],
                        input : [join_paths(meson.source_root(),'blocklib/blocks/multiply_const/blockname_templated_domain.hpp.j2'),
                                 yml_file],
                        output : ['multiply_const_cpu.hpp'],
                        command : [python3, join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_template.py'),'--input_file', '@INPUT0@', '--output_file', '@OUTPUT@', '--yaml_file', '@INPUT1@'])

gen_multiply_const_cpu_h = declare_dependency(sources : [gen_multiply_const_cpu_h])

blocks_multiply_const_deps = [gen_multiply_const_h, gen_multiply_const_cpu_h]