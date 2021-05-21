scheduler_mt_pybind_sources = files([
'scheduler_mt_pybind.cc'
 ] )

newsched_scheduler_mt_pybind = py3_mod.extension_module('scheduler_mt_python',
    scheduler_mt_pybind_sources, 
    dependencies : [newsched_runtime_dep, newsched_scheduler_mt_dep, python3_dep, pybind11_dep],
    link_language : 'cpp'
)

newsched_runtime_pybind_dep = declare_dependency(
					   link_with : newsched_scheduler_mt_pybind)