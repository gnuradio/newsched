blocks_head_files = files('head_cpu.cpp')

# Copy the hpp file to the build directory
gen_head_h = custom_target('gen_head_cpu_h',
                        input : ['head.hpp'],
                        output : ['head.hpp'],
                        command : ['python3', join_paths(meson.source_root(),'utils/copyfile.py'),
                            '--input_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--output_dir', join_paths(meson.build_root(), 'blocklib', 'blocks', 'include', 'gnuradio', 'blocks')],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_head_h = declare_dependency(sources : [gen_head_h])

blocks_head_deps = [gen_head_h]