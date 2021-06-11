subdir('include/gnuradio/analog')

analog_sources = []
analog_cu_sources = []
analog_pybind_sources = []
analog_pybind_names = []
analog_deps = []
# Individual block subdirectories
subdir('agc')

subdir('lib')
if (get_option('enable_python'))
    subdir('python/analog')
endif

subdir('test')
