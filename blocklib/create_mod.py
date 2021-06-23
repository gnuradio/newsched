# Python script to create new module in newsched

import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Create a new module within blocklib using newmod template')
parser.add_argument('create_mod', metavar='mod_name', type=str,
                    nargs='?', help='create new module given provided module name')

args = parser.parse_args()
mod_name = args.create_mod

# get current working directory
path = os.getcwd()

# copy newmod template
src = os.path.join(path, 'newmod')
dest = os.path.join(path, mod_name)

new_mod_dir = shutil.copytree(src, dest)

# change all occurances of newmod to mod_name in the new module directory

# include dir
include_dir = os.path.join('include/gnuradio', mod_name)
old_include = os.path.join(new_mod_dir, 'include/gnuradio/newmod')
new_include = os.path.join(new_mod_dir, include_dir)
os.rename(old_include, new_include)

# python dir
python_dir = os.path.join('python', mod_name)
old_python_dir = os.path.join(new_mod_dir, 'python/newmod')
new_python_dir = os.path.join(new_mod_dir, python_dir)
os.rename(old_python_dir, new_python_dir)

# change all occurances of newmod in files to mod_name

for path, subdirs, files in os.walk(new_mod_dir):
    for name in files:
        file_path = os.path.join(path, name)
        # read original file
        fin = open(file_path, "rt")
        data = fin.read()
        # replace all occurances of newmod with mod_name
        data = data.replace('newmod', mod_name)
        fin.close()
        # overwrite original file
        fin = open(file_path, "wt")
        fin.write(data)
        fin.close()

"""
# import the os module
import os
import sys

# get current working directory
path = os.getcwd()
print("The current working directory is %s" % path)

# get name of new module
mod_name = sys.argv[1]

# create new module
path = os.path.join(path, mod_name)
try:
    os.mkdir(path)
except OSError:
    print("Creation of the module %s failed" % mod_name)
else:
    print("Successfully created the module %s" % mod_name)

# create subdirectories in new module
directories = ['include/gnuradio/'+mod_name, 'lib', 'python/'+mod_name, 'test']
for directory in directories:
    try:
        os.makedirs(os.path.join(path, directory))
    except OSError:
        print("Creation of the directory %s failed" % directory)
    else:
        print("Successfully created the directory %s" % directory)

# add meson.builds to all directories
"""

