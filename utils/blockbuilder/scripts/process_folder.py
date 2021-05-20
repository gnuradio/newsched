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
    print("blockdir is " + blockdir)


    paths = []
    paths.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','templates'))
    paths.append(os.path.dirname(os.path.realpath(args.yaml_file)))
    env = Environment(loader = FileSystemLoader(paths))

    
    blockname = os.path.basename(os.path.dirname(os.path.realpath(args.yaml_file)))
    

    with open(args.yaml_file) as file:
        d = yaml.load(file, Loader=yaml.FullLoader)
        # Does this block specify a templated version
        templated = False
        if [x for x in d['properties'] if x['id'] == 'type']:
            templated = True 

        blockname_h = os.path.join(args.build_dir, 'blocklib', d['module'], blockname, blockname + '.hh')
        blockname_h_includedir = os.path.join(args.build_dir, 'blocklib', d['module'], 'include', 'gnuradio', d['module'], blockname + '.hh')
        full_outputfile = os.path.join(args.build_dir, args.output_file)

        if templated:
            template = env.get_template('blockname_templated.hh.j2')
        else:
            template = env.get_template('blockname.hh.j2')

        rendered = template.render(d)
        with open(blockname_h, 'w') as file:
            print("generating " + blockname_h)
            file.write(rendered)

        # Copy to the include dir
        shutil.copyfile(blockname_h, blockname_h_includedir)
        # shutil.copyfile(blockname_h, full_outputfile)

        # for impl in d['implementations']:
        #     if templated:
        #         template = env.get_template('blockname_templated_domain.hh.j2')
        #     else:
        #         template = env.get_template('blockname_domain.hh.j2')

        #     domain = impl['id']
        #     rendered = template.render(d, domain=domain)

        #     blockname_domain_h = os.path.join(args.build_dir, 'blocklib', d['module'], blockname, blockname + '_' + domain + '.hh')
        #     with open(blockname_domain_h, 'w') as file:
        #         print("generating " + blockname_domain_h)
        #         file.write(rendered)

if __name__ == "__main__":
    main()
