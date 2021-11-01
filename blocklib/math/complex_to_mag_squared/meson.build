srcs = files(['complex_to_mag_squared_cpu.cc'])

# if USE_CUDA
#     srcs += files('complex_to_mag_squared_cuda.cc')
#     # math_cu_sources += files('complex_to_mag_squared_cuda.cu')
# endif

gen_srcs = custom_target('gen_complex_to_mag_squared_h',
                        input : ['complex_to_mag_squared.yml'],
                        output : ['complex_to_mag_squared.cc', 'complex_to_mag_squared.hh', 'complex_to_mag_squared_pybind.cc', 'math_complex_to_mag_squared.block.yml'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output-cc', '@OUTPUT0@', 
                            '--output-hh', '@OUTPUT1@', 
                            '--output-pybind', '@OUTPUT2@', 
                            '--output-grc', '@OUTPUT3@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : [false, 'include/gnuradio/math',false,'share/gnuradio/grc/blocks'])

math_deps += declare_dependency(sources : [gen_srcs[1]] ) 
math_sources += [srcs, gen_srcs[0]]

if get_option('enable_python')                           
    math_pybind_sources += gen_srcs[2]
    math_pybind_names += 'complex_to_mag_squared'
endif
