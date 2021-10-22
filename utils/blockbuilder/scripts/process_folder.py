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
        if ('typekeys' in d and len(d['typekeys']) > 0):
            templated = len(d['typekeys'])


        blockname_h = os.path.join(args.build_dir, 'blocklib', d['module'], blockname, blockname + '.hh')
        blockname_h_includedir = os.path.join(args.build_dir, 'blocklib', d['module'], 'include', 'gnuradio', d['module'], blockname + '.hh')
        # full_outputfile = os.path.join(args.build_dir, args.output_file)

        if (args.output_file.endswith('.hh') ):
            if templated == 1:
                template = env.get_template('blockname_templated.hh.j2')
            elif templated == 2:
                template = env.get_template('blockname_templated2.hh.j2')
            else:
                template = env.get_template('blockname.hh.j2')

            rendered = template.render(d)
            with open(blockname_h, 'w') as file:
                print("generating " + blockname_h + " with " + str(templated))
                file.write(rendered)

            # Copy to the include dir
            shutil.copyfile(blockname_h, blockname_h_includedir)                

        else:
            blockname_cc = os.path.join(args.build_dir, 'blocklib', d['module'], blockname, blockname + '.cc')
            if templated == 1:
                template = env.get_template('blockname_templated.cc.j2')
            elif templated == 2:
                template = env.get_template('blockname_templated2.cc.j2')
            else:
                template = env.get_template('blockname.cc.j2')
            rendered = template.render(d)
            with open(blockname_cc, 'w') as file:
                print("generating " + blockname_cc)
                file.write(rendered)

        # copy the yaml file to the build dir
        yaml_path = os.path.join(args.build_dir,'grc','blocks')
        if not os.path.exists(yaml_path):
            os.makedirs(yaml_path)
        gr_blocks_path = os.path.join(yaml_path, os.path.basename(args.yaml_file))
        shutil.copyfile(args.yaml_file, gr_blocks_path)

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
