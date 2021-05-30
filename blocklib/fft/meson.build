fftw_dep = dependency('fftw3f')

subdir('include/gnuradio/fft')



fft_sources = []
fft_cu_sources = []
fft_pybind_sources = []
fft_pybind_names = []
fft_deps = []
# Individual block subdirectories
subdir('fft')

subdir('lib')
if (get_option('enable_python'))
    subdir('python/fft')
endif
subdir('test')