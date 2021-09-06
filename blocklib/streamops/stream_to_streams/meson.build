streamops_stream_to_streams_files = files(['stream_to_streams_cpu.cc'])

if cuda_dep.found() and get_option('enable_cuda')
    streamops_stream_to_streams_files += files('stream_to_streams_cuda.cc')
    # streamops_cu_sources += files('stream_to_streams_cuda.cu')
endif

gen_stream_to_streams_h = custom_target('gen_stream_to_streams_h',
                        input : ['stream_to_streams.yml'],
                        output : ['stream_to_streams.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/streamops')

gen_stream_to_streams_cc = custom_target('gen_stream_to_streams_cc',
                        input : ['stream_to_streams.yml'],
                        output : ['stream_to_streams.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/streamops')

streamops_deps += declare_dependency(sources : [gen_stream_to_streams_h] ) 
streamops_sources += [streamops_stream_to_streams_files, gen_stream_to_streams_cc]

if get_option('enable_python')
    gen_stream_to_streams_pybind = custom_target('gen_stream_to_streams_cpu_pybind',
                            input : ['stream_to_streams.yml'],
                            output : ['stream_to_streams_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)   
                            
    streamops_pybind_sources += gen_stream_to_streams_pybind
    streamops_pybind_names += 'stream_to_streams'
endif
