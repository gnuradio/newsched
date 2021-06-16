subdir('include/gnuradio/fileio')

fileio_sources = []
fileio_cu_sources = []
fileio_pybind_sources = []
fileio_pybind_names = []
fileio_deps = []
# Individual block subdirectories
subdir('file_source')
subdir('file_sink')

subdir('lib')
if (get_option('enable_python'))
    subdir('python/fileio')
endif

subdir('test')
