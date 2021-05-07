blocks_multiply_const_files = files('multiply_const_cpu.cpp')

if cuda_dep.found() and get_option('enable_cuda')
    blocks_multiply_const_files += files('multiply_const_cuda.cpp')
    blocks_multiply_const_cu_files = files('multiply_const_cuda.cu')
endif

# Copy the hpp file to the build directory
gen_multiply_const_h = custom_target('gen_multiply_const_cpu_h',
                        input : ['multiply_const.hpp'],
                        output : ['multiply_const.hpp'],
                        command : ['python3', join_paths(meson.source_root(),'utils/copyfile.py'),
                            '--input_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--output_dir', join_paths(meson.build_root(), 'blocklib', 'blocks', 'include', 'gnuradio', 'blocks')],
                        install : false,
                        install_dir : 'include/gnuradio/blocks')

gen_multiply_const_h = declare_dependency(sources : [gen_multiply_const_h])

blocks_multiply_const_deps = [gen_multiply_const_h]