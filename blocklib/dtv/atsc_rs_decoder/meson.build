yml_file = 'atsc_rs_decoder.yml'

dtv_atsc_rs_decoder_files = files(['atsc_rs_decoder_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     dtv_atsc_rs_decoder_files += files('atsc_rs_decoder_cuda.cc')
#     dtv_cu_sources += files('atsc_rs_decoder_cuda.cu')
# endif

gen_atsc_rs_decoder_h = custom_target('gen_atsc_rs_decoder_cpu_h',
                        input : ['atsc_rs_decoder.yml'],
                        output : ['atsc_rs_decoder.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/dtv')

gen_atsc_rs_decoder_cc = custom_target('gen_atsc_rs_decoder_cpu_cc',
                        input : ['atsc_rs_decoder.yml'],
                        output : ['atsc_rs_decoder.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

dtv_deps += declare_dependency(sources : [gen_atsc_rs_decoder_h])
dtv_sources += [dtv_atsc_rs_decoder_files, gen_atsc_rs_decoder_cc]

if get_option('enable_python')
    gen_atsc_rs_decoder_pybind = custom_target('gen_atsc_rs_decoder_cpu_pybind',
                            input : ['atsc_rs_decoder.yml'],
                            output : ['atsc_rs_decoder_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    dtv_pybind_sources += gen_atsc_rs_decoder_pybind
    dtv_pybind_names += 'atsc_rs_decoder'
endif
