subdir('include/gnuradio/streamops')

streamops_sources = []
streamops_cu_sources = []
streamops_pybind_sources = []
streamops_pybind_names = []
streamops_deps = []

# Individual block subdirectories
subdir('interleaved_short_to_complex')
subdir('lib')

if (get_option('enable_python'))
    subdir('python/streamops')
endif

subdir('test')