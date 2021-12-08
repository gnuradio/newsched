


subdir('blocks')
subdir('scripts')

install_subdir(
  'core',
  install_dir         : join_paths(py3_inst.get_install_dir(),'newsched','grc'),
  strip_directory     : false,
)

install_subdir(
  'converter',
  install_dir         : join_paths(py3_inst.get_install_dir(),'newsched','grc'),
  strip_directory     : false,
)

install_subdir(
  'gui',
  install_dir         : join_paths(py3_inst.get_install_dir(),'newsched','grc'),
  strip_directory     : false,
)

# Target for pure python
py3_inst.install_sources([
    '__init__.py',
    '__main__.py',
    'compiler.py',
    'main.py',
], 
subdir : join_paths('newsched','grc'))
