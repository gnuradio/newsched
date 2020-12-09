# SPDX-FileCopyrightText: 2020 Josh Morman 
#

project('newsched', 'cpp', 
  version : '0.0.0',
  license : 'GPLv3',
  default_options : ['cpp_std=c++17'])

cuda_available = add_languages('cuda',required : false)

boost_dep = dependency('boost', modules : ['program_options'], version : '>=1.65')
volk_dep = dependency('volk', version : '>=2.2')
yaml_dep = dependency('yaml-cpp', version : '>=0.6')
spdlog_dep = dependency('spdlog')
threads_dep = dependency('threads')
fmt_dep = dependency('fmt')
catch_dep = declare_dependency(include_directories : [join_paths('deps','catch','include')])
gtest_dep = dependency('gtest', main : true, version : '>=1.10', required : cuda_available and get_option('enable_testing'))
# gmock_dep = dependency('gmock', main : true, version : '>=1.10', required : get_option('enable_testing'))
cuda_dep = dependency('cuda', version : '>=11', modules : ['cufft'], required : cuda_available and get_option('enable_cuda'))

## Not sure yet how to search for zmq.hpp as dependency (header only, no pkgconfig)
zmq_dep = dependency('libzmq')

flatc = find_program('flatc', version : '>=1.12')

# subdir('deps/catch/include')
subdir('pmt')
subdir('runtime')
subdir('blocklib')
subdir('schedulers')
# subdir('bench')
subdir('docs')

# pkg_mod = import('pkgconfig')
# pkg_mod.generate(libraries : newsched,
#                  version : '0.1',
#                  name : 'libnewsched',
#                  filebase : 'newsched',
#                  description : 'Prototype of a modular GNU Radio')
