yml_file = 'agc.yml'

analog_agc_files = files(['agc_cpu.cc'])

gen_agc_h = custom_target('gen_agc_cpu_h',
                        input : ['agc.yml'],
                        output : ['agc.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/analog')

gen_agc_cc = custom_target('gen_agc_cpu_cc',
                        input : ['agc.yml'],
                        output : ['agc.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

analog_deps += declare_dependency(sources : [gen_agc_h])
analog_sources += [analog_agc_files, gen_agc_cc]

if get_option('enable_python')
    gen_agc_pybind = custom_target('gen_agc_cpu_pybind',
                            input : ['agc.yml'],
                            output : ['agc_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    analog_pybind_sources += gen_agc_pybind
    analog_pybind_names += 'agc'
endif
