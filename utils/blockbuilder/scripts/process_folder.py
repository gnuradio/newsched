from jinja2 import Template, FileSystemLoader, DictLoader, Environment
import os
import yaml
import argparse

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

        blockname_h = os.path.join(args.build_dir, 'blocklib', d['module'], '_include', 'gnuradio', d['module'], blockname + '.hpp')

        template = env.get_template('blockname_templated.hpp.j2')
        rendered = template.render(d)
        with open(blockname_h, 'w') as file:
            print("generating " + blockname_h)
            file.write(rendered)

        for impl in d['implementations']:
            template = env.get_template('blockname_templated_domain.hpp.j2')
            domain = impl['id']
            rendered = template.render(d, domain=domain)

            print("build: " + args.build_dir)
            blockname_domain_h = os.path.join(args.build_dir, 'blocklib', d['module'], blockname, blockname + '_' + domain + '.hpp')
            print("generating " + blockname_domain_h)
            with open(blockname_domain_h, 'w') as file:
                print("generating " + blockname_domain_h)
                file.write(rendered)

if __name__ == "__main__":
    main()
