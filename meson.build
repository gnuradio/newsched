# SPDX-FileCopyrightText: 2020 Josh Morman 
#

project('newsched', 'cpp', 
  version : '0.0.0',
  license : 'GPLv3',
  default_options : ['cpp_std=c++17'])

cc = meson.get_compiler('cpp')
rt_dep = cc.find_library('rt', required : false)
libdl_dep = cc.find_library('dl')

c_available = add_languages('c', required : true)
cuda_available = add_languages('cuda', required : false)
USE_CUDA = cuda_available and get_option('enable_cuda')

boost_dep = dependency('boost', modules : ['system','program_options','thread'], version : '>=1.65')
volk_dep = dependency('volk', version : '>=2.2')
yaml_dep = dependency('yaml-cpp', version : '>=0.6')
fmt_dep = dependency('fmt', method: 'cmake', modules: ['fmt::fmt'])
spdlog_dep = dependency('spdlog', method: 'cmake', modules : ['spdlog::spdlog'])
python3_dep = dependency('python3', required : get_option('enable_python'))
soapysdr_dep = dependency('SoapySDR', version : '>=0.7', required : false)
ENABLE_GR_SOAPY = soapysdr_dep.found() and get_option('enable_gr_soapy')

# Import python3 meson module which is used to find the Python dependencies.
py3_inst = import('python').find_installation('python3')

pybind11_dep = dependency('pybind11', required : get_option('enable_python'))
threads_dep = dependency('threads')
gtest_dep = dependency('gtest', main : true, version : '>=1.10', required : get_option('enable_testing'))
cuda_dep = dependency('cuda', version : '>=10.1', required : USE_CUDA, modules : ['cufft'])

qt5_mod = import('qt5', required: false)
qt5widgets_dep = dependency('qt5', modules : 'Widgets', required: false)
qwt_dep = dependency('Qwt', method: 'cmake', cmake_module_path: join_paths(meson.source_root(), 'cmake','Modules'), required: false )

if USE_CUDA
cusp_dep = dependency('cusp', version : '>=0.0.1', required : USE_CUDA) 
endif

pmtf_dep = dependency('pmtf')

cmake = import('cmake')

if (get_option('enable_testing'))
    TEST_ENV = environment()
    TEST_ENV.prepend('LD_LIBRARY_PATH', 
      join_paths( meson.build_root(),'schedulers','nbt','lib'),
      join_paths( meson.build_root(),'runtime','lib'),
      join_paths( meson.build_root(),'blocklib','analog','lib'),
      join_paths( meson.build_root(),'blocklib','blocks','lib'),
      join_paths( meson.build_root(),'blocklib','digital','lib'),
      join_paths( meson.build_root(),'blocklib','fec','lib'),
      join_paths( meson.build_root(),'blocklib','fft','lib'),
      join_paths( meson.build_root(),'blocklib','fileio','lib'),
      join_paths( meson.build_root(),'blocklib','filter','lib'),
      join_paths( meson.build_root(),'blocklib','math','lib'),
      join_paths( meson.build_root(),'blocklib','qtgui','lib'),
      join_paths( meson.build_root(),'blocklib','soapy','lib'),
      join_paths( meson.build_root(),'blocklib','streamops','lib')
    )
    TEST_ENV.prepend('PYTHONPATH', join_paths(meson.build_root(),'python'))

endif

subdir('runtime')
subdir('blocklib')
subdir('schedulers')
subdir('docs')
subdir('utils')
if (get_option('enable_python'))
  subdir('python/newsched')
  subdir('grc')
endif

