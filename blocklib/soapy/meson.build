soapy_sources = []
soapy_cu_sources = []
soapy_pybind_sources = []
soapy_pybind_names = []
soapy_deps = []

subdir('include/gnuradio/soapy')

# Individual block subdirectories
subdir('source')

subdir('lib')

if (get_option('enable_python'))
    subdir('python/soapy')
endif

if (get_option('enable_testing'))
    subdir('test')
endif