subdir('include/gnuradio/blocks')

blocks_sources = []
blocks_cu_sources = []
blocks_pybind_sources = []
blocks_pybind_names = []
blocks_deps = []
# Individual block subdirectories
subdir('copy')
subdir('annotator')
subdir('head')
subdir('load')
subdir('msg_forward')
subdir('multiply_const')
subdir('nop')
subdir('nop_head')
subdir('nop_source')
subdir('null_sink')
subdir('null_source')
subdir('vector_sink')
subdir('vector_source')


subdir('lib')
if (get_option('enable_python'))
    subdir('python/blocks')
endif
