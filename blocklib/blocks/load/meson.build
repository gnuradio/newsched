yml_file = 'load.yml'

blocks_load_files = files(['load_cpu.cc'])

if cuda_dep.found() and get_option('enable_cuda')
    blocks_load_files += files('load_cuda.cc')
    blocks_cu_sources += files('load_cuda.cu')
endif

gen_load_h = custom_target('gen_load_cpu_h',
                        input : ['load.yml'],
                        output : ['load.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_load_cc = custom_target('gen_load_cpu_cc',
                        input : ['load.yml'],
                        output : ['load.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

blocks_deps += declare_dependency(sources : [gen_load_h])
blocks_sources += [blocks_load_files, gen_load_cc]

if get_option('enable_python')
    gen_load_pybind = custom_target('gen_load_cpu_pybind',
                            input : ['load.yml'],
                            output : ['load_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    blocks_pybind_sources += gen_load_pybind
    blocks_pybind_names += 'load'
endif
