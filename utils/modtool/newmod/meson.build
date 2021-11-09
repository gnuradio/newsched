
###############################################################################
# If this is an OOT module, the following top level project things are needed
###############################################################################
project('ns-newmod', 'cpp', 
  version : '0.0.0',
  license : 'GPLv3',
  default_options : ['cpp_std=c++17'])

python3_dep = dependency('python3', required : get_option('enable_python'))
py3_inst = import('python').find_installation('python3')
pybind11_dep = dependency('pybind11', required : get_option('enable_python'))

prefix = get_option('prefix')
# Escape spaces
prefix_parts = prefix.split(' ')
prefix = '\ '.join(prefix_parts)
share_dir = join_paths(prefix,'share')
###############################################################################
# Else In-Tree module
###############################################################################
# share_dir = meson.project_source_root()

newmod_sources = []
newmod_cu_sources = []
newmod_pybind_sources = []
newmod_pybind_names = []
newmod_deps = []

volk_dep = dependency('volk', version : '>=2.2')
fmt_dep = dependency('fmt', method: 'cmake', modules: ['fmt::fmt'])
pmtf_dep = dependency('pmtf', version : '>= 0.0.2')
newsched_runtime_dep = dependency('newsched-runtime')

# Individual block subdirectories

subdir('blocklib/newmod')

