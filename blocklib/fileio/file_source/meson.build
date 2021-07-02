yml_file = 'file_source.yml'

fileio_file_source_files = files(['file_source_cpu.cc'])

gen_file_source_h = custom_target('gen_file_source_cpu_h',
                        input : ['file_source.yml'],
                        output : ['file_source.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/fileio')

gen_file_source_cc = custom_target('gen_file_source_cpu_cc',
                        input : ['file_source.yml'],
                        output : ['file_source.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

fileio_deps += declare_dependency(sources : [gen_file_source_h])
fileio_sources += [fileio_file_source_files, gen_file_source_cc]

if get_option('enable_python')
    gen_file_source_pybind = custom_target('gen_file_source_cpu_pybind',
                            input : ['file_source.yml'],
                            output : ['file_source_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    fileio_pybind_sources += gen_file_source_pybind
    fileio_pybind_names += 'file_source'
endif
