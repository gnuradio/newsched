soapy_sources += ['block_impl.cc']
soapy_deps += [newsched_runtime_dep, soapysdr_dep, volk_dep, fmt_dep, pmtf_dep]

block_cpp_args = ['-DHAVE_CPU']

incdir = include_directories(['../include/gnuradio/soapy','../include'])
newsched_blocklib_soapy_lib = library('newsched-blocklib-soapy', 
    soapy_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : soapy_deps,
    cpp_args : block_cpp_args)

newsched_blocklib_soapy_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_soapy_lib,
                       dependencies : soapy_deps)
