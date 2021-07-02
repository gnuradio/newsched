yml_file = 'atsc_sync.yml'

dtv_atsc_sync_files = files(['atsc_sync_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     dtv_atsc_sync_files += files('atsc_sync_cuda.cc')
#     dtv_cu_sources += files('atsc_sync_cuda.cu')
# endif

gen_atsc_sync_h = custom_target('gen_atsc_sync_cpu_h',
                        input : ['atsc_sync.yml'],
                        output : ['atsc_sync.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/dtv')

gen_atsc_sync_cc = custom_target('gen_atsc_sync_cpu_cc',
                        input : ['atsc_sync.yml'],
                        output : ['atsc_sync.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

dtv_deps += declare_dependency(sources : [gen_atsc_sync_h])
dtv_sources += [dtv_atsc_sync_files, gen_atsc_sync_cc]

if get_option('enable_python')
    gen_atsc_sync_pybind = custom_target('gen_atsc_sync_cpu_pybind',
                            input : ['atsc_sync.yml'],
                            output : ['atsc_sync_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    dtv_pybind_sources += gen_atsc_sync_pybind
    dtv_pybind_names += 'atsc_sync'
endif
