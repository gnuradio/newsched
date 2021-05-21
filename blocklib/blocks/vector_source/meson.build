blocks_vector_source_files = files('vector_source_cpu.cc')

# Copy the hpp file to the build directory
gen_vector_source_h = custom_target('gen_vector_source_cpu_h',
                        input : ['vector_source.hh'],
                        output : ['vector_source.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/copyfile.py'),
                            '--input_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--output_dir', join_paths(meson.build_root(), 'blocklib', 'blocks', 'include', 'gnuradio', 'blocks')],
                        install : false,
                        install_dir : 'include/gnuradio/blocks')

gen_vector_source_h = declare_dependency(sources : [gen_vector_source_h])

blocks_vector_source_deps = [gen_vector_source_h]

if get_option('enable_python')
    blocks_pybind_sources += files('vector_source_pybind.cc')
endif