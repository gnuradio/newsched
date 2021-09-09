scheduler_nbt_pybind_sources = files([
'scheduler_nbt_pybind.cc'
 ] )

newsched_scheduler_nbt_pybind = py3_inst.extension_module('scheduler_nbt_python',
    scheduler_nbt_pybind_sources, 
    dependencies : [newsched_runtime_dep, newsched_scheduler_nbt_dep, python3_dep, pybind11_dep],
    link_language : 'cpp',
    install : true,
    install_dir : join_paths(py3_inst.get_install_dir(),'newsched','schedulers','nbt')
)

newsched_runtime_pybind_dep = declare_dependency(
					   link_with : newsched_scheduler_nbt_pybind)