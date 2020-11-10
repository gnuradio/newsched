# SPDX-FileCopyrightText: 2020 Josh Morman 
#

project('newsched', 'cpp',
  version : '0.0.0',
  license : 'GPLv3',
  default_options : ['cpp_std=c++17'])

boost_dep = dependency('boost', modules : ['program_options'], version : '>=1.65')
volk_dep = dependency('volk', version : '>=2.2')
yaml_dep = dependency('yaml-cpp', version : '>=0.6')
spdlog_dep = dependency('spdlog')
threads_dep = dependency('threads')
catch_dep = declare_dependency(include_directories : [join_paths('deps','catch','include')])

## Not sure yet how to search for ZMQ as dependency (header only, no pkgconfig)
# zmq_dep = dependency('cppzmq')
# zmq_dep = dependency('ZeroMQ', method : 'cmake')

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