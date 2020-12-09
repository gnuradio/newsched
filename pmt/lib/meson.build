pmt_sources = ['pmt.cpp']

incdir = include_directories('../include')
newsched_pmt_lib = library('newsched_pmt', pmt_sources, include_directories : incdir, install : true)

newsched_pmt_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_pmt_lib)

# pmtf_sources = ['pmtf.cpp', pmt_gen_h]
# newsched_pmtf_lib = library('newsched_pmtf', pmtf_sources, include_directories : incdir, install : true)

# newsched_pmtf_dep = declare_dependency(include_directories : incdir,
# 					   link_with : newsched_pmtf_lib)

pmtf_sources = [ pmt_gen_h, 'pmtf.cpp', 'pmtf_scalar.cpp','pmtf_vector.cpp','pmtf_string.cpp', 'pmtf_map.cpp']
newsched_pmtf_lib = library('newsched_pmtf', pmtf_sources, include_directories : incdir, install : true)

newsched_pmtf_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_pmtf_lib)