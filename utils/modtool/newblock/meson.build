yml_file = 'newblock.yml'

newmod_newblock_files = files(['newblock_cpu.cc'])


if cuda_dep.found() and get_option('enable_cuda')
    newmod_newblock_files += files('newblock_cuda.cc')
    newmod_cu_sources += files('newblock_cuda.cu')
endif

gen_newblock_h = custom_target('gen_newblock_cpu_h',
                        input : ['newblock.yml'],
                        output : ['newblock.hh'],
                        command : ['python3', join_paths(share_dir,'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/newmod')

gen_newblock_cc = custom_target('gen_newblock_cpu_cc',
                        input : ['newblock.yml'],
                        output : ['newblock.cc'],
                        command : ['python3', join_paths(share_dir,'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

newmod_deps += declare_dependency(sources : [gen_newblock_h])
newmod_sources += [newmod_newblock_files, gen_newblock_cc]

if get_option('enable_python')
    gen_newblock_pybind = custom_target('gen_newblock_cpu_pybind',
                            input : ['newblock.yml'],
                            output : ['newblock_pybind.cc'],
                            command : ['python3', join_paths(share_dir,'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    newmod_pybind_sources += gen_newblock_pybind
    newmod_pybind_names += 'newblock'
endif
