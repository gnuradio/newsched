from jinja2 import Template, FileSystemLoader, DictLoader, Environment
import os
import yaml
import argparse
import shutil

def argParse():
    """Parses commandline args."""
    desc='Scrape the doxygen generated xml for docstrings to insert into python bindings'
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument("--yaml_file")
    parser.add_argument("--output_file")
    parser.add_argument("--build_dir")

    return parser.parse_args()

def main():
    args = argParse()
    # env = Environment(loader = FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','templates'))

    # Set FileSystemLoader path to be dir of yml file and dir of this script

    blockdir = os.path.dirname(os.path.realpath(args.yaml_file))
    # print("blockdir is " + blockdir)


    paths = []
    paths.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','templates'))
    paths.append(os.path.dirname(os.path.realpath(args.yaml_file)))
    env = Environment(loader = FileSystemLoader(paths))

    
    blockname = os.path.basename(os.path.dirname(os.path.realpath(args.yaml_file)))
    

    with open(args.yaml_file) as file:
        d = yaml.load(file, Loader=yaml.FullLoader)
        # Does this block specify a templated version
        templated = 0
        if [x for x in d['properties'] if x['id'] == 'type']:
            templated = 1 
        elif [x for x in d['properties'] if x['id'] == 'templates']:
            templated = 2


        filename = os.path.join(args.build_dir, 'blocklib', d['module'], blockname, blockname + '_pybind.cc')
        # full_outputfile = os.path.join(args.build_dir, args.output_file)

        if templated == 1:
            template = env.get_template('blockname_pybind_templated.cc.j2')
        elif templated == 2:
            template = env.get_template('blockname_pybind_templated2.cc.j2')
        else:
            template = env.get_template('blockname_pybind.cc.j2')

        rendered = template.render(d)
        with open(filename, 'w') as file:
            print("generating " + filename)
            file.write(rendered)

if __name__ == "__main__":
    main()
