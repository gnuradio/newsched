#!/usr/bin/env python3

import re
import sys
from argparse import ArgumentParser

parser = ArgumentParser('Parse Perf Sched')

parser.add_argument('-w', '--what', default='migrations',
                    help=(f'what to parse (migrations, schedules, debug) default=%(default)s)'))


args = parser.parse_args()
what = args.what

re_migrate = re.compile(' *(.+) *(\d+) \[(\d+)\] *(\d+\.\d+): sched:sched_migrate_task: comm=(.+) pid=(\d+) prio=(\d+) orig_cpu=(\d+) dest_cpu=(\d+)$')

re_sched = re.compile(' *(.+) (\d+) \[(\d+)\] *(\d+\.\d+): sched:sched_switch: prev_comm=(.+) prev_pid=(\d+) prev_prio=(\d+) prev_state=(.+) ==> next_comm(.+) next_pid=(\d+) next_prio=(\d+)$')


if what == 'migrations':
    print('time,cpu_from,cpu_to,thread')

if what == 'resched':
    print('time,thread,state')

for l in sys.stdin:
    m = re_migrate.match(l)
    n = re_sched.match(l)
    if m:
        if what == 'migrations':
            print(f'{m[4]},{m[8]},{m[9]},{m[5]}')
    elif n:
        if what == 'resched':
            print(f'{n[4]},{n[5]},{n[8]}')
    else:
        if what == 'debug':
            print('no match:', l)

