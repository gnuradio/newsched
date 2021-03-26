copy_h = configure_file(copy: true, input : ['../../../copy/copy.hpp'], output : 'copy.hpp')

blocks_headers = [
    copy_h,
    'fanout.hpp',
    'head.hpp',
    'multiply_const.hpp',
    'null_sink.hpp',
    'null_source.hpp',
    'vector_sink.hpp',
    'vector_source.hpp',
]

install_headers(blocks_headers, subdir : 'gnuradio/blocks')
