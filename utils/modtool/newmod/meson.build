newmod_sources = []
newmod_cu_sources = []
newmod_pybind_sources = []
newmod_pybind_names = []
newmod_deps = []

subdir('include/gnuradio/newmod')

# Individual block subdirectories


subdir('lib')

if (get_option('enable_python'))
    subdir('python/newmod')
endif

if (get_option('enable_testing'))
    subdir('test')
endif