blocks_vector_sink_files = files('vector_sink_cpu.cc')

# Copy the hpp file to the build directory
gen_vector_sink_h = custom_target('gen_vector_sink_cpu_h',
                        input : ['vector_sink.hh'],
                        output : ['vector_sink.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/copyfile.py'),
                            '--input_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--output_dir', join_paths(meson.build_root(), 'blocklib', 'blocks', 'include', 'gnuradio', 'blocks')],
                        install : false,
                        install_dir : 'include/gnuradio/blocks')

gen_vector_sink_h = declare_dependency(sources : [gen_vector_sink_h])

blocks_vector_sink_deps = [gen_vector_sink_h]