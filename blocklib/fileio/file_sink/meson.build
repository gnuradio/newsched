yml_file = 'file_sink.yml'

fileio_file_sink_files = files(['file_sink_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     fileio_file_sink_files += files('file_sink_cuda.cc')
#     fileio_cu_sources += files('file_sink_cuda.cu')
# endif

gen_file_sink_h = custom_target('gen_file_sink_cpu_h',
                        input : ['file_sink.yml'],
                        output : ['file_sink.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/fileio')

gen_file_sink_cc = custom_target('gen_file_sink_cpu_cc',
                        input : ['file_sink.yml'],
                        output : ['file_sink.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

fileio_deps += declare_dependency(sources : [gen_file_sink_h])
fileio_sources += [fileio_file_sink_files, gen_file_sink_cc]

if get_option('enable_python')
    gen_file_sink_pybind = custom_target('gen_file_sink_cpu_pybind',
                            input : ['file_sink.yml'],
                            output : ['file_sink_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    fileio_pybind_sources += gen_file_sink_pybind
    fileio_pybind_names += 'file_sink'
endif
