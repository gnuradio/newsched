run_command('python3', join_paths(meson.project_source_root(),'utils','blockbuilder','scripts','gen_meson.py'), 
  join_paths(meson.project_source_root(),'blocklib'), check: true)

subdir('math')
subdir('blocks')
subdir('analog')
subdir('fec')
subdir('fft')
subdir('filter')
subdir('streamops')
subdir('digital')
subdir('fileio')
subdir('qtgui')
if (ENABLE_GR_SOAPY)
  subdir('soapy')
endif
