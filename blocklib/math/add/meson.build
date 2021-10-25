srcs = files(['add_cpu.cc'])

if cuda_dep.found() and get_option('enable_cuda')
    srcs += files('add_cuda.cc')
    # math_cu_sources += files('add_cuda.cu')
endif

gen_srcs = custom_target('gen_add_h',
                        input : ['add.yml'],
                        output : ['add.cc', 'add.hh', 'add_pybind.cc', 'math_add.block.yml'],
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
    math_pybind_names += 'add'
endif
