math_sources += [
    'fast_atan2f.cc'
]

math_deps += [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep]


block_cpp_args = ['-DHAVE_CPU']
 if cuda_dep.found() and get_option('enable_cuda')
     block_cpp_args += '-DHAVE_CUDA'

     math_deps += [cuda_dep, cusp_dep]

endif

incdir = include_directories(['../include/gnuradio/math','../include'])
newsched_blocklib_math_lib = library('newsched-blocklib-math', 
    math_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : math_deps,
    cpp_args : block_cpp_args)

newsched_blocklib_math_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_math_lib,
                       dependencies : math_deps)
