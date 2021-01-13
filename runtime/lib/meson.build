incdir = [
    '../include',
]

runtime_deps = [yaml_dep, spdlog_dep, threads_dep, fmt_dep]

runtime_sources = [
  'parameter_types.cpp',
  'edge.cpp',
  'graph.cpp',
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
