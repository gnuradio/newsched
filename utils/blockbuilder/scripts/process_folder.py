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
    parser.add_argument("--output-cc")
    parser.add_argument("--output-hh")
    parser.add_argument("--output-pybind")
    parser.add_argument("--output-grc", nargs='+')
    parser.add_argument("--grc-index", nargs='+')
    parser.add_argument("--build_dir")

    return parser.parse_args()

def is_list(value):
    return isinstance(value, list)

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
    env.filters['is_list'] = is_list

    
    blockname = os.path.basename(os.path.dirname(os.path.realpath(args.yaml_file)))
    
  

    with open(args.yaml_file) as file:
        d = yaml.load(file, Loader=yaml.FullLoader)
        # Does this block specify a templated version
        templated = 0
        if ('typekeys' in d and len(d['typekeys']) > 0):
            templated = len(d['typekeys'])

        if (args.output_hh ):
            blockname_h = os.path.join(args.build_dir, 'blocklib', d['module'], blockname, os.path.basename(args.output_hh))
            blockname_h_includedir = os.path.join(args.build_dir, 'blocklib', d['module'], 'include', 'gnuradio', d['module'], os.path.basename(args.output_hh))
            if templated >= 1:
                template = env.get_template('blockname_templated.h.j2')
            else:
                template = env.get_template('blockname.h.j2')

            rendered = template.render(d)
            with open(blockname_h, 'w') as file:
                print("generating " + blockname_h)
                file.write(rendered)

            # Copy to the include dir
            shutil.copyfile(blockname_h, blockname_h_includedir)                

        if args.output_cc:
            blockname_cc = os.path.join(args.build_dir, 'blocklib', d['module'], blockname, os.path.basename(args.output_cc))
            if templated >= 1:
                template = env.get_template('blockname_templated.cc.j2')
            else:
                template = env.get_template('blockname.cc.j2')
            rendered = template.render(d)
            with open(blockname_cc, 'w') as file:
                print("generating " + blockname_cc)
                file.write(rendered)

        if args.output_pybind:
            filename = os.path.join(args.build_dir, 'blocklib', d['module'], blockname, os.path.basename(args.output_pybind))

            if templated >= 1:
                template = env.get_template('blockname_pybind_templated.cc.j2')
            else:
                template = env.get_template('blockname_pybind.cc.j2')

            rendered = template.render(d)
            with open(filename, 'w') as file:
                print("generating " + filename)
                file.write(rendered)

        if args.output_grc:
            template = env.get_template('blockname.grc.j2')
            idx = 0
            if 'grc_multiple' in d:
                # TODO - handle the grc-idx and impl
                for grc_file in d['grc_multiple']:
                    filename = os.path.join(args.build_dir, 'blocklib', d['module'], blockname, os.path.basename(args.output_grc[idx]))
                    rendered = template.render(d, grc=grc_file)
                    with open(filename, 'w') as file:
                        print("generating " + filename)
                        file.write(rendered)
                    idx += 1
            else:
                for grcidx,fn in zip(args.grc_index,args.output_grc):
                    filename = os.path.join(args.build_dir, 'blocklib', d['module'], blockname, os.path.basename(fn))
                    rendered = template.render(d)
                    with open(filename, 'w') as file:
                        print("generating " + filename)
                        file.write(rendered)


        # copy the yaml file to the build dir
        yaml_path = os.path.join(args.build_dir,'grc','blocks')
        if not os.path.exists(yaml_path):
            os.makedirs(yaml_path)
        gr_blocks_path = os.path.join(yaml_path, os.path.basename(args.yaml_file))
        shutil.copyfile(args.yaml_file, gr_blocks_path)


if __name__ == "__main__":
    main()
