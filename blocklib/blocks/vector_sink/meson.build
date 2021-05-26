
blocks_vector_sink_files = files(['vector_sink_cpu.cc'])

# if cuda_dep.found() and get_option('enable_cuda')
#     blocks_vector_sink_files += files('vector_sink_cuda.cc')
#     blocks_vector_sink_cu_files = files('vector_sink_cuda.cu')
# endif

gen_vector_sink_h = custom_target('gen_vector_sink_h',
                        input : ['vector_sink.yml'],
                        output : ['vector_sink.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_vector_sink_cc = custom_target('gen_vector_sink_cc',
                        input : ['vector_sink.yml'],
                        output : ['vector_sink.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')


blocks_deps += declare_dependency(sources : [gen_vector_sink_h] ) 
blocks_sources += [blocks_vector_sink_files, gen_vector_sink_cc]

gen_vector_sink_pybind = custom_target('gen_vector_sink_cpu_pybind',
                        input : ['vector_sink.yml'],
                        output : ['vector_sink_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)                    

if get_option('enable_python')
    blocks_pybind_sources += gen_vector_sink_pybind
endif
