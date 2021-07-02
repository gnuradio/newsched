subdir('include/gnuradio/filter')

filter_sources = []
filter_cu_sources = []
filter_pybind_sources = []
filter_pybind_names = []
filter_deps = []

# Individual block subdirectories
subdir('dc_blocker')
subdir('moving_average')
subdir('lib')

if (get_option('enable_python'))
    subdir('python/filter')
endif

subdir('test')