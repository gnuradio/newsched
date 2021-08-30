newmod_deps += [newsched_runtime_dep, volk_dep, fmt_dep, pmt_dep]

block_cpp_args = ['-DHAVE_CPU']
# if cuda_dep.found() and get_option('enable_cuda')
#     block_cpp_args += '-DHAVE_CUDA'

#     newsched_blocklib_newmod_cu = library('newsched-blocklib-newmod-cu', 
#         newmod_cu_sources, 
#         include_directories : incdir, 
#         install : true, 
#         dependencies : [cuda_dep])

#     newsched_blocklib_newmod_cu_dep = declare_dependency(include_directories : incdir,
#                         link_with : newsched_blocklib_newmod_cu,
#                         dependencies : cuda_dep)

#     newmod_deps += [newsched_blocklib_newmod_cu_dep, cuda_dep]

# endif

incdir = include_directories(['../include/gnuradio/newmod','../include'])
newsched_blocklib_newmod_lib = library('newsched-blocklib-newmod', 
    newmod_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : newmod_deps,
    cpp_args : block_cpp_args)

newsched_blocklib_newmod_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_newmod_lib,
                       dependencies : newmod_deps)
