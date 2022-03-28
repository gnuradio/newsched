fftw_dep = dependency('fftw3f')

kernel_deps = [volk_dep, gnuradio_gr_dep, fftw_dep]
kernel_sources = [
    'analog/agc.cc',
    'fft/window.cc',
    'fft/fftw_fft.cc',
    'filter/fir_filter.cc',
    'filter/firdes.cc',
    'filter/mmse_fir_interpolator_ff.cc',
    'filter/moving_averager.cc',
    'filter/polyphase_filterbank.cc',
    'math/fast_atan2f.cc',
    'math/fxpt.cc',
    'math/random.cc'
]

compiler = meson.get_compiler('cpp')
cpp_args = []
code = '''#define _GNU_SOURCE
     #include <math.h>
     int main(){double x, sin, cos; sincos(x, &sin, &cos); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_SINCOS')
  cpp_args += '-DHAVE_SINCOS'
endif

code = '''#define _GNU_SOURCE
     #include <math.h>
     int main(){float x, sin, cos; sincosf(x, &sin, &cos); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_SINCOSF')
  cpp_args += '-DHAVE_SINCOSF'
endif

code = '''#include <math.h>
     int main(){sinf(0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_SINF')
  cpp_args += '-DHAVE_SINF'
endif

code = '''#include <math.h>
     int main(){cosf(0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_COSF')
  cpp_args += '-DHAVE_COSF'
endif


subdir('math')

incdir = include_directories(['../include'])
gr_kernel_lib = library('gnuradio-kernellib', 
    kernel_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : kernel_deps)

gr_kernel_lib_dep = declare_dependency(include_directories : incdir,
					   link_with : gr_kernel_lib,
                       dependencies : kernel_deps)
