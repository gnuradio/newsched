subdir('reed-solomon')

# TODO - export this as a subproject of newsched

conf = configuration_data()
conf.set('prefix', prefix)
conf.set('exec_prefix', '${prefix}')
conf.set('libdir', join_paths('${prefix}',get_option('libdir')))
conf.set('includedir', join_paths('${prefix}',get_option('includedir')))
conf.set('LIBVER', '0.0.0')

cmake_conf = configuration_data()
cmake_conf.set('libdir', join_paths(prefix,get_option('libdir')))
cmake_conf.set('module', 'fec')
cmake.configure_package_config_file(
  name : 'newsched-fec',
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
             name : 'libnewsched-fec',
             filebase : 'newsched-fec',
             install_dir : get_option('prefix') / 'lib' / 'pkgconfig',
             description : 'Newsched GR 4.0 Prototype')
