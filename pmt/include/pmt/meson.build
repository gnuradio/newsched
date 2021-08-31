files = [
    'api.h',
    'attributes.h',
    'pmt.h',
    'pmt_pool.h',
    'pmt_serial_tags.h',
    'pmt_sugar.h'
]
install_headers(files, subdir : 'pmt')

pmt_gen_h = custom_target('flatc-output',
                        input : ['pmt.fbs'],
                        output : ['pmt_generated.h'],
                        command : [flatc, '-c', '--reflect-types','--gen-mutable','--reflect-names','--cpp-std','c++17', '-o', '@OUTDIR@', '@INPUT@'])

pmt_gen_h_dep = declare_dependency(sources : [pmt_gen_h])