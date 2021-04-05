yml_file = 'copy.yml'

gen_copy_h = custom_target('gen_copy_cpu_h',
                        input : [yml_file],
                        output : ['copy.hpp'],
                        command : [python3, join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_copy_h = declare_dependency(sources : [gen_copy_h])

blocks_copy_deps = [gen_copy_h]