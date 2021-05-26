yml_file = 'copy.yml'

blocks_copy_files = files(['copy_cpu.cc'])


if cuda_dep.found() and get_option('enable_cuda')
    blocks_copy_files += files('copy_cuda.cc')
    blocks_copy_cu_files = files('copy_cuda.cu')
endif

gen_copy_h = custom_target('gen_copy_cpu_h',
                        input : ['copy.yml'],
                        output : ['copy.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_copy_cc = custom_target('gen_copy_cpu_cc',
                        input : ['copy.yml'],
                        output : ['copy.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

blocks_deps += declare_dependency(sources : [gen_copy_h])
blocks_sources += [blocks_copy_files, gen_copy_cc]

if get_option('enable_python')
    gen_copy_pybind = custom_target('gen_copy_cpu_pybind',
                            input : ['copy.yml'],
                            output : ['copy_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    blocks_pybind_sources += gen_copy_pybind
    blocks_pybind_names += 'copy'
endif
