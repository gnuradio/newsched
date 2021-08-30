yml_file = 'fft.yml'

fft_fft_files = files(['fft_cpu.cc'])
if cuda_dep.found() and get_option('enable_cuda')
    fft_fft_files += files('fft_cuda.cc')
    fft_cu_sources += files('fft_shift.cu')
endif

gen_fft_h = custom_target('gen_fft_cpu_h',
                        input : ['fft.yml'],
                        output : ['fft.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/fft')

gen_fft_cc = custom_target('gen_fft_cpu_cc',
                        input : ['fft.yml'],
                        output : ['fft.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

fft_deps += declare_dependency(sources : [gen_fft_h])
fft_sources += [fft_fft_files, gen_fft_cc]

if get_option('enable_python')
    gen_fft_pybind = custom_target('gen_fft_cpu_pybind',
                            input : ['fft.yml'],
                            output : ['fft_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    fft_pybind_sources += gen_fft_pybind
    fft_pybind_names += 'fft'
endif
