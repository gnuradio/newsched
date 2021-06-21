yml_file = 'atsc_deinterleaver.yml'

dtv_atsc_deinterleaver_files = files(['atsc_deinterleaver_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     dtv_atsc_deinterleaver_files += files('atsc_deinterleaver_cuda.cc')
#     dtv_cu_sources += files('atsc_deinterleaver_cuda.cu')
# endif

gen_atsc_deinterleaver_h = custom_target('gen_atsc_deinterleaver_cpu_h',
                        input : ['atsc_deinterleaver.yml'],
                        output : ['atsc_deinterleaver.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/dtv')

gen_atsc_deinterleaver_cc = custom_target('gen_atsc_deinterleaver_cpu_cc',
                        input : ['atsc_deinterleaver.yml'],
                        output : ['atsc_deinterleaver.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

dtv_deps += declare_dependency(sources : [gen_atsc_deinterleaver_h])
dtv_sources += [dtv_atsc_deinterleaver_files, gen_atsc_deinterleaver_cc]

if get_option('enable_python')
    gen_atsc_deinterleaver_pybind = custom_target('gen_atsc_deinterleaver_cpu_pybind',
                            input : ['atsc_deinterleaver.yml'],
                            output : ['atsc_deinterleaver_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    dtv_pybind_sources += gen_atsc_deinterleaver_pybind
    dtv_pybind_names += 'atsc_deinterleaver'
endif
