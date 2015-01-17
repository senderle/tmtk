#!/usr/bin/python

import argparse
import itertools
import re
import operator

def parse_composition_file(path, filename_rex=None):
    composition = {}
    with open(path, 'r') as cfile:
        for line in cfile:
            line = line.split('#')[0].strip() # discard comments, whitespace
            if not line:
                continue                      # drop empty lines 

            fields = line.split()
            filename = fields[1]

            if filename_rex is not None:
                filename_data = re.search(filename_rex, filename)
                if filename_data is None:
                    filename_data = {'docid':filename}
                else:
                    filename_data = filename_data.groupdict()
                    if 'docid' not in filename_data:
                        filename_data['docid'] = filename
            else:
                filename_data = {'docid':filename}

            filename_data['text_n'] = int(fields[0])
            topics = fields[2::2]
            proportions = fields[3::2]
            filename_data.update((int(t), float(p)) for t, p in 
                                 zip(topics, proportions))
            composition[filename_data['docid']] = filename_data
    return composition

def top_texts(texts, topics, n):
    ordered = sorted(texts)
    vectors = [[texts[tx][tp] for tx in ordered] for tp in topics]
    multiplied = [reduce(operator.mul, col) for col in zip(*vectors)]
    proportions_texts = sorted(zip(multiplied, ordered), reverse=True)[0:n]
    return proportions_texts

def add_metadata(comp, md_file):
    with open(md_file) as md:
        for line in md:
            firstline = line.strip()
            if firstline:
                break

        if firstline[0] == '#':
            fieldnames = firstline[1:].split()
        else:
            fieldnames = [str(n) for n, _ in enumerate(firstline.split())]
            md = itertools.chain((firstline,), md)

        fieldnames = fieldnames[1:]
        for line in md:
            fields = line.split('\t')
            tid = fields[0]
            if tid in comp:
                comp[tid].update(zip(fieldnames, fields[1:]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse a MALLET composition '
        'file and print the top texts for a given topic or topics')
    parser.add_argument('-n', '--num-texts', type=int, default=20, 
        metavar='integer', help='The number of texts to display.')
    parser.add_argument('-m', '--metadata-file', type=str, metavar='filename',
        help='A file containing metadata in a tab-delimited table, with '
        'the file id in the first column. Typically, the file id will simply '
        'be the filename, but see the help for the `--parser-rex` option for '
        'other possibilities.')
    parser.add_argument('-r', '--parser-rex', type=str,
        metavar='regular_expression', help='A regular expression for '
        'extracting metadata from filenames as listed in the MALLET '
        'composition file. Only named groups will be captured. For example, '
        'the named group `(?P<year>\d\d\d\d)` will capture a four-digit '
        'sequence and associate it with the key `year`. To use a customized '
        'file id in your metadata file, create a group named `docid`.')
    parser.add_argument('-f', '--field-to-print', type=str, action='append',
        metavar='name', help='A metadata field to include in the output. '
        'May be used multiple times.')
    parser.add_argument('composition_file', type=str, help='A composition '
        'file produced by MALLET (via the --output-doc-topics option).')
    parser.add_argument('topic_num', type=int, nargs='+', help='A list of '
        'topics to compare, specified by the topic number assigned by '
        'MALLET.')

    args = parser.parse_args()
    texts = parse_composition_file(args.composition_file, args.parser_rex)
    if args.metadata_file is not None:
        add_metadata(texts, args.metadata_file)
    
    fields = args.field_to_print
    if 'docid' not in fields:
        fields.append('docid')
    for p, t in top_texts(texts, args.topic_num, args.num_texts):
        print p, 
        for f in fields:
            print texts[t][f],
        print
        


