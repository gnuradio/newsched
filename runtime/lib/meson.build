runtime_sources = [
  'parameter_types.cpp',
  'block.cpp',
  'sync_block.cpp',
#   domain_adapter_zmq.cpp,
  'edge.cpp',
  'graph.cpp',
  'graph_utils.cpp',
  'flat_graph.cpp',
  'flowgraph_monitor.cpp',
  'flowgraph.cpp',
  'logging.cpp',
  'scheduler.cpp',
  'realtime.cpp',
  'pagesize.cpp',
  'vmcirc_buffer.cpp'
]

incdir = [
    '../include',
]

newsched_runtime_lib = library('newsched-runtime', 
    runtime_sources, 
    install : true, 
    dependencies : [yaml_dep, spdlog_dep],
    include_directories : incdir)

newsched_runtime_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_runtime_lib)
