sources = [
    'file_source.cpp',
    'file_sink_base.cpp',
    'file_sink.cpp'
]

incdir = include_directories(['../include/gnuradio/fileio','../include'])
newsched_blocklib_fileio_lib = library('newsched-blocklib-fileio', 
    sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep])

newsched_blocklib_fileio_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_fileio_lib)
