# pmt_sources = ['pmt.cc']

incdir = include_directories('../include')


pmtf_sources = [
	'pmtf.cc', 
	'pmtf_scalar.cc',
	'pmtf_vector.cc',
	'pmtf_string.cc', 
	'pmtf_map.cc'
]
newsched_pmtf_lib = library('newsched-pmtf', pmtf_sources, include_directories : incdir, install : true, dependencies : [pmt_gen_h_dep])

pmtf_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_pmtf_lib)

pmt_sources = [
	'old_pmt.cc',
	'pmt_unv.cc',
	'pmt_serialize.cc',
	'pmt_io.cc',
	'pmt_pool.cc'
]
newsched_pmt_lib = library('newsched-pmt', pmt_sources, include_directories : incdir, install : true, dependencies : [boost_dep])


pmt_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_pmt_lib)


# TODO - export this as a subproject of newsched

prefix = get_option('prefix')
# Escape spaces
prefix_parts = prefix.split(' ')
prefix = '\ '.join(prefix_parts)

conf = configuration_data()
conf.set('prefix', prefix)
conf.set('exec_prefix', '${prefix}')
conf.set('libdir', join_paths('${prefix}',get_option('libdir')))
conf.set('includedir', join_paths('${prefix}',get_option('includedir')))
conf.set('LIBVER', '0.0.0')

cmake_conf = configuration_data()
cmake_conf.set('libdir', join_paths(prefix,get_option('libdir')))
cmake.configure_package_config_file(
  name : 'newsched-pmt',
  input : join_paths(meson.source_root(),'cmake','Modules','newschedConfigPmt.cmake.in'),
  install_dir : get_option('prefix') / 'lib' / 'cmake' / 'newsched',
  configuration : cmake_conf
)

pkg = import('pkgconfig')
libs = []     # the library/libraries users need to link against
h = ['.'] # subdirectories of ${prefix}/${includedir} to add to header path
pkg.generate(libraries : libs,
             subdirs : h,
             version : meson.project_version(),
             name : 'libnewsched-pmt',
             filebase : 'newsched-pmt',
             install_dir : get_option('prefix') / 'lib' / 'pkgconfig',
             description : 'Newsched Runtime')

