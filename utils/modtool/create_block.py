# Python script to create new block within a module

import argparse
import fnmatch
import os
import shutil
import sys

# Argument Parser Setup
parser = argparse.ArgumentParser(description='Create a new block within specified module using newblock template')
parser.add_argument('create_block', metavar='block_name', type=str, help='name of new block')
parser.add_argument('mod_name', metavar='mod_name', type=str, help='name of module block will be inserted in, module must already exist')
parser.add_argument('--cpu', action='store_true', help='indicate whether cpu files will be generated')
parser.add_argument('--cuda', action='store_true', help='indicate whether cuda files will be generated')
parser.add_argument('--templated', action='store_true', help='indicate whether files will be templated')

args = parser.parse_args()
block_name = args.create_block # name of new block to be created
mod_name = args.mod_name # name of module where new block will be created
cpu_arg = args.cpu # boolean for cpu files
cuda_arg = args.cuda # boolean for cuda files
templated_arg = args.templated # boolean for templated

# get current working directory
path = os.getcwd()
path_of_this_file = os.path.dirname(os.path.abspath(__file__))

# check if module exists, if not return an error
mod_path = os.path.join(path, mod_name)
mod_exists = os.path.isdir(mod_path)
if not mod_exists:
    sys.exit('Error: module ' + mod_name + ' does not exist')

# copy newblock template
src = os.path.join(path_of_this_file, 'newblock')
dest = os.path.join(mod_path, block_name)
new_block_dir = shutil.copytree(src,dest)

# if cpu is not specified, delete all cpu files
if not cpu_arg:
    files = os.listdir(new_block_dir)
    for file in files:
        if 'cpu' in file:
            os.remove(os.path.join(new_block_dir, file))

# if cuda is not specified, delete all cuda files
if not cuda_arg:
    files = os.listdir(new_block_dir)
    for file in files:
        if 'cuda' in file:
            os.remove(os.path.join(new_block_dir, file))

# if templated is specified, remove non-templated files and rename files
# else, remove templated files
if templated_arg:
    files = os.listdir(new_block_dir)
    for file in files:
        if '_t.' not in file:
            if file == 'meson.build': # leave meson.build as is
                continue
            else:
                os.remove(os.path.join(new_block_dir, file))
    files = os.listdir(new_block_dir)
    for file in files:
        if '_t.' in file:
            s = file
            index = s.index('.')
            s = s[0 : index-2] + s[index :]
            os.rename(os.path.join(new_block_dir, file), os.path.join(new_block_dir, s))
else:
    files = os.listdir(new_block_dir)
    for file in files:
        if '_t.' in file:
            os.remove(os.path.join(new_block_dir, file))

# rename all files from newblock to block_name
files = os.listdir(new_block_dir)
for file in files:
    if 'newblock' in file:
        new_file_name = file.replace('newblock', block_name)
        os.rename(os.path.join(new_block_dir, file), os.path.join(new_block_dir, new_file_name))

# change all occurences of newblock and newmod in files to block_name and mod_name respectively
files = os.listdir(new_block_dir)
for name in files:
    file_path = os.path.join(new_block_dir, name)
    # read original file
    fin = open(file_path, "rt")
    data = fin.read()
    # replace all occurances of newmod with mod_name and newblock with block_name
    data = data.replace('newmod', mod_name)
    data = data.replace('newblock', block_name)
    fin.close()
    # overwrite original file
    fin = open(file_path, "wt")
    fin.write(data)
    fin.close()