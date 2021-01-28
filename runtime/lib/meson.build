incdir = [
    '../include',
]

runtime_deps = [yaml_dep, spdlog_dep, threads_dep, fmt_dep]

if cuda_dep.found() and get_option('enable_cuda')
  cu_sources = [
    'cudabuffer.cu',
    'cudabuffer_pinned.cu'
  ]

  newsched_runtime_cu_lib = library('newsched-runtime-cu',
      cu_sources, 
      cuda_args: ['-DTAG_IS_SHARED=1', '-DTAG_IS_BUILDING=1'],
      install : true, 
      # dependencies : [cuda_dep],
      include_directories : incdir)

  newsched_runtime_cu_dep = declare_dependency(include_directories : incdir,
              link_with : newsched_runtime_cu_lib)

  runtime_deps += [newsched_runtime_cu_dep]
endif

runtime_sources = [
  'parameter_types.cpp',
  'edge.cpp',
  'graph.cpp',
  'graph_utils.cpp',
  'flat_graph.cpp',
  'flowgraph_monitor.cpp',
  'flowgraph.cpp',
  'logging.cpp'
]

newsched_runtime_lib = library('newsched-runtime', 
    runtime_sources, 
    install : true, 
    dependencies : runtime_deps,
    link_language : 'cpp',
    include_directories : incdir)

newsched_runtime_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_runtime_lib)
