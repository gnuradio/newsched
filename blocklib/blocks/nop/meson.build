yml_file = 'nop.yml'

blocks_nop_files = files(['nop_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     blocks_nop_files += files('nop_cuda.cc')
#     blocks_nop_cu_files = files('nop_cuda.cu')
# endif

gen_nop_h = custom_target('gen_nop_cpu_h',
                        input : ['nop.yml'],
                        output : ['nop.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_nop_cc = custom_target('gen_nop_cpu_cc',
                        input : ['nop.yml'],
                        output : ['nop.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

             

blocks_deps += declare_dependency(sources : [gen_nop_h])
blocks_sources += [blocks_nop_files, gen_nop_cc]

if get_option('enable_python')
gen_nop_pybind = custom_target('gen_nop_cpu_pybind',
                        input : ['nop.yml'],
                        output : ['nop_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)       
    blocks_pybind_sources += gen_nop_pybind
endif
