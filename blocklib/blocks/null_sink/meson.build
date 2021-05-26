yml_file = 'null_sink.yml'

blocks_null_sink_files = files(['null_sink_cpu.cc'])


if cuda_dep.found() and get_option('enable_cuda')
    # blocks_null_sink_files += files('null_sink_cuda.cc')
    # blocks_null_sink_cu_files = files('null_sink_cuda.cu')
endif

gen_null_sink_h = custom_target('gen_null_sink_cpu_h',
                        input : ['null_sink.yml'],
                        output : ['null_sink.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_null_sink_cc = custom_target('gen_null_sink_cpu_cc',
                        input : ['null_sink.yml'],
                        output : ['null_sink.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

blocks_deps += declare_dependency(sources : [gen_null_sink_h])
blocks_sources += [blocks_null_sink_files, gen_null_sink_cc]

if get_option('enable_python')
    gen_null_sink_pybind = custom_target('gen_null_sink_cpu_pybind',
                            input : ['null_sink.yml'],
                            output : ['null_sink_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)    
    blocks_pybind_sources += gen_null_sink_pybind
    blocks_pybind_names += 'null_sink'
endif
