math_sources = []
math_cu_sources = []
math_pybind_sources = []
math_pybind_names = []
math_deps = []

subdir('include/gnuradio/math')

subdir('conjugate')
subdir('complex_to_mag_squared')
subdir('multiply')
subdir('divide')


subdir('lib')

if (get_option('enable_python'))
    subdir('python/math')
endif

if (get_option('enable_testing'))
    subdir('test')
endif
