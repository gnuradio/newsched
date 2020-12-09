pmt_gen_h = custom_target('flatc-output',
                        input : ['pmt.fbs'],
                        output : ['pmt_generated.h'],
                        command : [flatc, '-c', '--reflect-types','--gen-mutable','--reflect-names','--cpp-std','c++17', '-o', '@OUTDIR@/../include/pmt', '@INPUT@'])
