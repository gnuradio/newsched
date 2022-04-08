digital_deps += [gnuradio_gr_dep, volk_dep, fmt_dep, pmtf_dep, gr_kernel_lib_dep]

digital_sources += []

block_cpp_args = ['-DHAVE_CPU']
# if cuda_dep.found() and get_option('enable_cuda')
#     block_cpp_args += '-DHAVE_CUDA'

#     gnuradio_blocklib_digital_cu = library('gnuradio-blocklib-digital-cu', 
#         digital_cu_sources, 
#         include_directories : incdir, 
#         install : true, 
#         dependencies : [cuda_dep])

#     gnuradio_blocklib_digital_cu_dep = declare_dependency(include_directories : incdir,
#                         link_with : gnuradio_blocklib_digital_cu,
#                         dependencies : cuda_dep)

#     digital_deps += [gnuradio_blocklib_digital_cu_dep, cuda_dep]

# endif

incdir = include_directories(['../include/gnuradio/digital','../include'])
gnuradio_blocklib_digital_lib = library('gnuradio-blocklib-digital', 
    digital_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : digital_deps,
    cpp_args : block_cpp_args)

gnuradio_blocklib_digital_dep = declare_dependency(include_directories : incdir,
					   link_with : gnuradio_blocklib_digital_lib,
                       dependencies : digital_deps)

# TODO - export this as a subproject of gnuradio

conf = configuration_data()
conf.set('prefix', prefix)
conf.set('exec_prefix', '${prefix}')
conf.set('libdir', join_paths('${prefix}',get_option('libdir')))
conf.set('includedir', join_paths('${prefix}',get_option('includedir')))
conf.set('LIBVER', '0.0.1')

cmake_conf = configuration_data()
cmake_conf.set('libdir', join_paths(prefix,get_option('libdir')))
cmake_conf.set('module', 'digital')
cmake.configure_package_config_file(
  name : 'gnuradio-digital',
  input : join_paths(meson.source_root(),'cmake','Modules','gnuradioConfigModule.cmake.in'),
  install_dir : get_option('prefix') / 'lib' / 'cmake' / 'gnuradio',
  configuration : cmake_conf
)

pkg = import('pkgconfig')
libs = []     # the library/libraries users need to link against
h = ['.'] # subdirectories of ${prefix}/${includedir} to add to header path
pkg.generate(libraries : libs,
             subdirs : h,
             version : meson.project_version(),
             name : 'libgnuradio-digital',
             filebase : 'gnuradio-digital',
             install_dir : get_option('prefix') / 'lib' / 'pkgconfig',
             description : 'GNU Radio Digital Blocks Library')
