zeromq_headers = [
    'api.h'
]

install_headers(zeromq_headers, subdir : 'gnuradio/zeromq')

# TODO - export this as a subproject of newsched

conf = configuration_data()
conf.set('prefix', prefix)
conf.set('exec_prefix', '${prefix}')
conf.set('libdir', join_paths('${prefix}',get_option('libdir')))
conf.set('includedir', join_paths('${prefix}',get_option('includedir')))
conf.set('LIBVER', '0.0.1')

cmake_conf = configuration_data()
cmake_conf.set('libdir', join_paths(prefix,get_option('libdir')))
cmake_conf.set('module', 'zeromq')
cmake.configure_package_config_file(
  name : 'newsched-zeromq',
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
             name : 'libnewsched-zeromq',
             filebase : 'newsched-zeromq',
             install_dir : get_option('prefix') / 'lib' / 'pkgconfig',
             description : 'Newsched Digital Library')
