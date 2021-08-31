sources = [
    'moving_averager.cc',
    'fir_filter.cc',
    'mmse_fir_interpolator_ff.cc',
    'polyphase_filterbank.cc',
    'firdes.cc'
]

filter_sources += sources
filter_deps += [newsched_runtime_dep, newsched_blocklib_fft_dep, volk_dep, fmt_dep, pmt_dep]

block_cpp_args = ['-DHAVE_CPU']
if USE_CUDA
    block_cpp_args += '-DHAVE_CUDA'

#     newsched_blocklib_filter_cu = library('newsched-blocklib-filter-cu', 
#         filter_cu_sources, 
#         include_directories : incdir, 
#         install : true, 
#         dependencies : [cuda_dep])

#     newsched_blocklib_filter_cu_dep = declare_dependency(include_directories : incdir,
#                         link_with : newsched_blocklib_filter_cu,
#                         dependencies : cuda_dep)

    filter_deps += [cuda_dep, cusp_dep]
endif


incdir = include_directories(['../include/gnuradio/filter','../include'])
newsched_blocklib_filter_lib = library('newsched-blocklib-filter', 
    filter_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : filter_deps,
    link_args : '-lcusp',  # why is this necesary???
    cpp_args : block_cpp_args)

newsched_blocklib_filter_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_filter_lib,
                       dependencies : filter_deps)

# TODO - export this as a subproject of newsched

conf = configuration_data()
conf.set('prefix', prefix)
conf.set('exec_prefix', '${prefix}')
conf.set('libdir', join_paths('${prefix}',get_option('libdir')))
conf.set('includedir', join_paths('${prefix}',get_option('includedir')))
conf.set('LIBVER', '0.0.0')

cmake_conf = configuration_data()
cmake_conf.set('libdir', join_paths(prefix,get_option('libdir')))
cmake_conf.set('module', 'filter')
cmake.configure_package_config_file(
  name : 'newsched-filter',
  input : join_paths(meson.source_root(),'cmake','Modules','newschedConfigModule.cmake.in'),
  install_dir : get_option('prefix') / 'lib' / 'cmake' / 'newsched',
  configuration : cmake_conf
)

pkg = import('pkgconfig')
libs = []     # the library/libraries users need to link against
h = ['.'] # subdirectories of ${prefix}/${includedir} to add to header path
pkg.generate(libraries : libs,
             subdirs : h,
             version : meson.project_version(),
             name : 'libnewsched-filter',
             filebase : 'newsched-filter',
             install_dir : get_option('prefix') / 'lib' / 'pkgconfig',
             description : 'Newsched GR 4.0 Prototype')
