yml_file = 'copy.yml'

blocks_copy_files = files('copy_cpu.cpp')

if cuda_dep.found() and get_option('enable_cuda')
    blocks_copy_files += files('copy_cuda.cpp')
    blocks_copy_cu_files = files('copy_cuda.cu')
endif

# Copy the hpp file to the build directory
# gen_copy_h = custom_target('gen_copy_cpu_h',
#                         input : ['copy.hpp'],
#                         output : ['copy.hpp'],
#                         command : ['python3', join_paths(meson.source_root(),'utils/copyfile.py'),
#                             '--input_file', '@INPUT@', 
#                             '--output_file', '@OUTPUT@', 
#                             '--output_dir', join_paths(meson.build_root(), 'blocklib', 'blocks', 'include', 'gnuradio', 'blocks')],
#                         install : true,
#                         install_dir : 'include/gnuradio/blocks')

gen_copy_h = custom_target('gen_copy_cpu_h',
                        input : ['copy.yml'],
                        output : ['copy.hpp'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_copy_h = declare_dependency(sources : [gen_copy_h])

blocks_copy_deps = [gen_copy_h]