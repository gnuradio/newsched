yml_file = 'freq_sink.yml'

qtgui_freq_sink_files = files(['freq_sink_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     qtgui_freq_sink_files += files('freq_sink_cuda.cc')
#     qtgui_cu_sources += files('freq_sink_cuda.cu')
# endif

gen_freq_sink_h = custom_target('gen_freq_sink_cpu_h',
                        input : ['freq_sink.yml'],
                        output : ['freq_sink.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/qtgui')

gen_freq_sink_cc = custom_target('gen_freq_sink_cpu_cc',
                        input : ['freq_sink.yml'],
                        output : ['freq_sink.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

qtgui_deps += declare_dependency(sources : [gen_freq_sink_h])
qtgui_sources += [qtgui_freq_sink_files, gen_freq_sink_cc]

if get_option('enable_python')
    gen_freq_sink_pybind = custom_target('gen_freq_sink_cpu_pybind',
                            input : ['freq_sink.yml'],
                            output : ['freq_sink_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    qtgui_pybind_sources += gen_freq_sink_pybind
    qtgui_pybind_names += 'freq_sink'
endif
