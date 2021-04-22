blocks_annotator_files = files('annotator_cpu.cpp')

# Copy the hpp file to the build directory
gen_annotator_h = custom_target('gen_annotator_cpu_h',
                        input : ['annotator.hpp'],
                        output : ['annotator.hpp'],
                        command : ['python3', join_paths(meson.source_root(),'utils/copyfile.py'),
                            '--input_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--output_dir', join_paths(meson.build_root(), 'blocklib', 'blocks', 'include', 'gnuradio', 'blocks')],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_annotator_h = declare_dependency(sources : [gen_annotator_h])

blocks_annotator_deps = [gen_annotator_h]