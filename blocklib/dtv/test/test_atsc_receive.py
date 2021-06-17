#!/usr/bin/env python3

from newsched import gr, streamops, fileio, dtv, analog, filter
import argparse

def argParse():
    """Parses commandline args."""
    desc='Scrape the doxygen generated xml for docstrings to insert into python bindings'
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument("filename")

    return parser.parse_args()

def main():
    args = argParse()

    sps = 1.1
    atsc_sym_rate = 4.5e6 / 286 * 684
    oversampled_rate = atsc_sym_rate * sps

    fg = gr.flowgraph()

    src = fileio.file_source(2*gr.sizeof_short, args.filename, False)
    is2c = streamops.interleaved_short_to_complex(False, 32768.0)
    fpll = dtv.atsc_fpll(oversampled_rate)
    dcb = filter.dc_blocker_ff(4096, True)
    agc = analog.agc_ff(1e-5, 4.0, 1.0)
    sync = dtv.atsc_sync(oversampled_rate)
    fschk = dtv.atsc_fs_checker()
    eq = dtv.atsc_equalizer()
    vit = dtv.atsc_viterbi_decoder()
    dei = dtv.atsc_deinterleaver()
    rsd = dtv.atsc_rs_decoder()
    der = dtv.atsc_derandomizer()
    snk = fileio.file_sink(gr.sizeof_char*188, "/tmp/mpeg.live.ts")
    # snk = fileio.file_sink(gr.sizeof_float*832, "/tmp/mpeg.live.ts")

    fg.connect(src, 0, is2c, 0)
    fg.connect(is2c, 0, fpll, 0)
    fg.connect(fpll, 0, dcb, 0)
    fg.connect(dcb, 0, agc, 0)
    fg.connect(agc, 0, sync, 0)
    
    fg.connect(sync, 0, fschk, 0)
    fg.connect(fschk, 0, eq, 0)
    fg.connect(fschk, 1, eq, 1)
    fg.connect(eq, 0, vit, 0)
    fg.connect(eq, 1, vit, 1)
    fg.connect(vit, 0, dei, 0)
    fg.connect(vit, 1, dei, 1)
    fg.connect(dei, 0, rsd, 0)
    fg.connect(dei, 1, rsd, 1)
    fg.connect(rsd, 0, der, 0)
    fg.connect(rsd, 1, der, 1)
    fg.connect(der, 0, snk, 0)

    # fg.connect(src, 0, is2c, 0)
    # fg.connect(is2c, 0, fpll, 0)
    # fg.connect(fpll, 0, dcb, 0)
    # fg.connect(dcb, 0, agc, 0)
    # fg.connect(agc, 0, sync, 0)
    # fg.connect(sync, 0, snk, 0)

    fg.start()
    fg.wait()

if __name__ == "__main__":
    main()
  
