yml_file = 'time_sink.yml'

qtgui_time_sink_files = files(['time_sink_cpu.cc'])


if USE_CUDA
    # qtgui_time_sink_files += files('time_sink_cuda.cc')
    # qtgui_cu_sources += files('time_sink_cuda.cu')
endif

gen_time_sink_h = custom_target('gen_time_sink_cpu_h',
                        input : ['time_sink.yml'],
                        output : ['time_sink.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/qtgui')

gen_time_sink_cc = custom_target('gen_time_sink_cpu_cc',
                        input : ['time_sink.yml'],
                        output : ['time_sink.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

qtgui_deps += declare_dependency(sources : [gen_time_sink_h])
qtgui_sources += [qtgui_time_sink_files, gen_time_sink_cc]

if get_option('enable_python')
    gen_time_sink_pybind = custom_target('gen_time_sink_cpu_pybind',
                            input : ['time_sink.yml'],
                            output : ['time_sink_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    qtgui_pybind_sources += gen_time_sink_pybind
    qtgui_pybind_names += 'time_sink'
endif
