# SPDX-FileCopyrightText: 2020 Josh Morman 
#

project('newsched', 'cpp', 
  version : '0.0.0',
  license : 'GPLv3',
  default_options : ['cpp_std=c++17'])

volk_dep = dependency('volk', version : '>=2.2')
yaml_dep = dependency('yaml-cpp', version : '>=0.6')
spdlog_dep = dependency('spdlog')
threads_dep = dependency('threads')
fmt_dep = dependency('fmt')
gtest_dep = dependency('gtest', main : true, version : '>=1.10', required : get_option('enable_testing'))

subdir('runtime')
subdir('blocklib')
subdir('schedulers')
subdir('docs')
