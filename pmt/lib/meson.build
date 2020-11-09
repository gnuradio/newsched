pmt_sources = ['pmt.cpp']

incdir = include_directories('../include')
newsched_pmt_lib = library('newsched_pmt', pmt_sources, include_directories : incdir, install : true)

newsched_pmt_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_pmt_lib)