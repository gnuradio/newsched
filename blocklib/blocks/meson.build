subdir('include/gnuradio/blocks')

blocks_sources = []
blocks_pybind_sources = []
blocks_deps = []
# Individual block subdirectories
subdir('copy')
subdir('annotator')
subdir('head')
subdir('multiply_const')
subdir('null_sink')
subdir('null_source')
subdir('vector_sink')
subdir('vector_source')


subdir('lib')
if (get_option('enable_python'))
    subdir('python')
endif
