#!/usr/bin/env python3

import sys
import re

re_probe = re.compile('^ *([a-z,0-9,_]+) *\d+ *\[(\d+)\] *(\d+\.\d+).* arg1=(-?\d+)$')

print("time,block,cpu,prod")

for l in sys.stdin:
    m = re_probe.match(l)
    if m:
        print(f'{m[3]},{m[1]},{int(m[2])},{m[4]}')
    else:
        sys.stderr.write('no match:' + l)

        
