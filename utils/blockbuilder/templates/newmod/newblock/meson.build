yml_file = 'newblock.yml'

blocks_newblock_files = files(['newblock_cpu.cc'])

gen_newblock_h = custom_target('gen_newblock_cpu_h',
                        input : ['newblock.yml'],
                        output : ['newblock.hh'],
                        command : ['python3', join_paths(SCRIPTS_DIR,'process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_newblock_cc = custom_target('gen_newblock_cpu_cc',
                        input : ['newblock.yml'],
                        output : ['newblock.cc'],
                        command : ['python3', join_paths(SCRIPTS_DIR,'process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

blocks_deps += declare_dependency(sources : [gen_newblock_h])
blocks_sources += [blocks_newblock_files, gen_newblock_cc]

if get_option('enable_python')
    gen_newblock_pybind = custom_target('gen_newblock_cpu_pybind',
                            input : ['newblock.yml'],
                            output : ['newblock_pybind.cc'],
                            command : ['python3', join_paths(SCRIPTS_DIR,'process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    blocks_pybind_sources += gen_newblock_pybind
    blocks_pybind_names += 'newblock'
endif
