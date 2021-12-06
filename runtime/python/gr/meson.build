
subdir('bindings')

srcs = [
    '__init__.py',
    'gateway_numpy.py'
    ]

if USE_CUDA
    srcs += 'gateway_cupy.py'
endif

foreach s: srcs
configure_file(copy: true,
    input: s,
    output: s
)
endforeach

py3_inst.install_sources(files(srcs), subdir : join_paths('newsched','gr'))
