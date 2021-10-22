subdir('include/gnuradio/blocks')

blocks_sources = []
blocks_cu_sources = []
blocks_pybind_sources = []
blocks_pybind_names = []
blocks_deps = []
# Individual block subdirectories
subdir('copy')
# subdir('delay')
# subdir('annotator')
# subdir('head')
# subdir('load')
# subdir('msg_forward')
# subdir('nop')
# subdir('nop_head')
# subdir('nop_source')
# subdir('null_sink')
# subdir('null_source')
# subdir('throttle')
# subdir('vector_sink')
# subdir('vector_source')


subdir('lib')
if (get_option('enable_python'))
    subdir('python/blocks')
endif

if (get_option('enable_testing'))
    subdir('test')
endif
