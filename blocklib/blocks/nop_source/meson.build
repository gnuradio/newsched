yml_file = 'nop_source.yml'

blocks_nop_source_files = files(['nop_source_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     # blocks_nop_source_files += files('nop_source_cuda.cc')
#     # blocks_nop_source_cu_files = files('nop_source_cuda.cu')
# endif

gen_nop_source_h = custom_target('gen_nop_source_cpu_h',
                        input : ['nop_source.yml'],
                        output : ['nop_source.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_nop_source_cc = custom_target('gen_nop_source_cpu_cc',
                        input : ['nop_source.yml'],
                        output : ['nop_source.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        


blocks_deps += declare_dependency(sources : [gen_nop_source_h])
blocks_sources += [blocks_nop_source_files, gen_nop_source_cc]

if get_option('enable_python')
gen_nop_source_pybind = custom_target('gen_nop_source_cpu_pybind',
                        input : ['nop_source.yml'],
                        output : ['nop_source_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)                    

    blocks_pybind_sources += gen_nop_source_pybind
endif
