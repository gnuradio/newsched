# pmt_sources = ['pmt.cpp']

incdir = include_directories('../include')


pmtf_sources = ['pmtf.cpp', 'pmtf_scalar.cpp','pmtf_vector.cpp','pmtf_string.cpp', 'pmtf_map.cpp']
newsched_pmtf_lib = library('newsched_pmtf', pmtf_sources, include_directories : incdir, install : true, dependencies : [pmt_gen_h_dep])

pmtf_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_pmtf_lib)
