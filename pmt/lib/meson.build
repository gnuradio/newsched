# pmt_sources = ['pmt.cc']

incdir = include_directories('../include')


pmtf_sources = [
	'pmtf.cc', 
	'pmtf_scalar.cc',
	'pmtf_vector.cc',
	'pmtf_string.cc', 
	'pmtf_map.cc'
]
newsched_pmtf_lib = library('newsched_pmtf', pmtf_sources, include_directories : incdir, install : true, dependencies : [pmt_gen_h_dep])

pmtf_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_pmtf_lib)

pmt_sources = [
	'old_pmt.cc',
	'pmt_unv.cc',
	'pmt_serialize.cc',
	'pmt_io.cc',
	'pmt_pool.cc'
]
newsched_pmt_lib = library('newsched_pmt', pmt_sources, include_directories : incdir, install : true, dependencies : [boost_dep])


pmt_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_pmt_lib)