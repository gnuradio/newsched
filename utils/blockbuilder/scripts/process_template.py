from jinja2 import Template, FileSystemLoader, DictLoader, Environment
import os
import yaml
import argparse

def argParse():
    """Parses commandline args."""
    desc='Scrape the doxygen generated xml for docstrings to insert into python bindings'
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")
    parser.add_argument("--yaml_file")

    return parser.parse_args()

def main():
    args = argParse()
    env = Environment(loader = FileSystemLoader(os.path.abspath('/')))

    template = env.get_template(args.input_file)

    with open(args.yaml_file) as file:
        d = yaml.load(file, Loader=yaml.FullLoader)

    rendered = template.render(d)

    with open(args.output_file, 'w') as file:
        file.write(rendered)

if __name__ == "__main__":
    main()
