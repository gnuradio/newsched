
yml_file = 'multiply_const.yml'

gen_multiply_const_h = custom_target('gen_multiply_const_cpu_h',
                        input : [yml_file],
                        output : ['multiply_const.hpp'],
                        command : [python3, join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_multiply_const_h = declare_dependency(sources : [gen_multiply_const_h])

blocks_multiply_const_deps = [gen_multiply_const_h]

