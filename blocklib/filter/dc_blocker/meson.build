filter_dc_blocker_files = files(['dc_blocker_cpu.cc'])

if cuda_dep.found() and get_option('enable_cuda')
    filter_dc_blocker_files += files('dc_blocker_cuda.cc')
    filter_cu_sources += files('dc_blocker_cuda.cu')
endif

gen_dc_blocker_h = custom_target('gen_dc_blocker_h',
                        input : ['dc_blocker.yml'],
                        output : ['dc_blocker.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/filter')

gen_dc_blocker_cc = custom_target('gen_dc_blocker_cc',
                        input : ['dc_blocker.yml'],
                        output : ['dc_blocker.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/filter')

filter_deps += declare_dependency(sources : [gen_dc_blocker_h] ) 
filter_sources += [filter_dc_blocker_files, gen_dc_blocker_cc]

if get_option('enable_python')
    gen_dc_blocker_pybind = custom_target('gen_dc_blocker_cpu_pybind',
                            input : ['dc_blocker.yml'],
                            output : ['dc_blocker_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)   
                            
    filter_pybind_sources += gen_dc_blocker_pybind
    filter_pybind_names += 'dc_blocker'
endif
