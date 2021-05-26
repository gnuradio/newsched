yml_file = 'annotator.yml'

blocks_annotator_files = files(['annotator_cpu.cc'])


if cuda_dep.found() and get_option('enable_cuda')
    # blocks_annotator_files += files('annotator_cuda.cc')
    # blocks_annotator_cu_files = files('annotator_cuda.cu')
endif

gen_annotator_h = custom_target('gen_annotator_cpu_h',
                        input : ['annotator.yml'],
                        output : ['annotator.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_annotator_cc = custom_target('gen_annotator_cpu_cc',
                        input : ['annotator.yml'],
                        output : ['annotator.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

gen_annotator_pybind = custom_target('gen_annotator_cpu_pybind',
                        input : ['annotator.yml'],
                        output : ['annotator_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)                    

blocks_deps += declare_dependency(sources : [gen_annotator_h])
blocks_sources += [blocks_annotator_files, gen_annotator_cc]

if get_option('enable_python')
    #blocks_pybind_sources += files('annotator_pybind.cc')
    blocks_pybind_sources += gen_annotator_pybind
endif
