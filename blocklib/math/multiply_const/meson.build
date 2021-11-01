srcs = files(['multiply_const_cpu.cc'])

if USE_CUDA
    srcs += files('multiply_const_cuda.cc')
    # math_cu_sources += files('multiply_const_cuda.cu')
endif

gen_srcs = custom_target('gen_multiply_const_h',
                        input : ['multiply_const.yml'],
                        output : ['multiply_const.cc', 'multiply_const.hh', 'multiply_const_pybind.cc', 'math_multiply_const.block.yml'],
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
    math_pybind_names += 'multiply_const'
endif
