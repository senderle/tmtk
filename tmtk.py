#!/usr/bin/python

import argparse
import itertools
import re
import operator
import numpy
import networkx
import networkx.readwrite.json_graph
import sys
import json

class FileType(object):
    """Factory for creating file object types

    Modified from the original argparse version to reject stdin/out so
    that it's always OK to close immediately after reading.

    Instances of FileType are typically passed as type= arguments to the
    ArgumentParser add_argument() method.

    Keyword Arguments:
        - mode -- A string indicating how the file is to be opened. Accepts the
            same values as the builtin open() function.
        - bufsize -- The file's desired buffer size. Accepts the same values as
            the builtin open() function.
    """

    def __init__(self, mode='r', bufsize=-1):
        self._mode = mode
        self._bufsize = bufsize

    def __call__(self, string):
        try:
            return open(string, self._mode, self._bufsize)
        except IOError as e:
            message = argparse._("can't open '%s': %s")
            raise argparse.ArgumentTypeError(message % (string, e))

    def __repr__(self):
        args = self._mode, self._bufsize
        args_str = ', '.join(repr(arg) for arg in args if arg != -1)
        return '%s(%s)' % (type(self).__name__, args_str)

def parse_composition_file(open_file, filename_rex=None):
    '''Accepts an already-open composition file produced by MALLET's
    --output-doc-topics option and creates a dictionary for each document.
    Each dictionary cotains at least a `docid` key, a `text_n` key, and an
    integer key for each topic in the model. The value of `docid` defaults 
    to the filename for each document reported by MALLET. If `filename_rex`
    is passed, it's used to parse the filename reported by mallet. Any named
    group creates a key; for example, the group `(?P<year>\d\d\d\d)` creates
    a `year` key in the dictionary. This can be used to override the `docid`
    key. However, the `text_n` key and integer keys cannot be overidden.
    The dictionaries are stored in another dictionary using `docid` values
    as keys, and that dictionary is returned.'''
    composition = {}
    with open_file as cfile:
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

def parse_metadata(open_md_file):
    with open_md_file as md:
        for line in md:
            firstline = line.strip()
            if firstline:
                break

        if firstline[0] == '#':
            fieldnames = firstline[1:].split('\t')
        else:
            fieldnames = ['field_' + str(n) for n, _x in
                          enumerate(firstline.split('\t'))]
            md = itertools.chain((firstline,), md)
    
        for line in md:
            fields = [f.strip() for f in line.split('\t')]
            yield zip(fieldnames, fields)

def add_text_metadata(comp, open_md_file):
    '''Accepts a dictionary of document dictionaries (as created by
    parse_composition_file()) and extends each of them with metadata from 
    a tab-delimited csv file. The first column of the metadata file is
    treated as the `docid` for purposes of matching rows to documents. Later
    columns are given field names based on the first line of the metadata
    file, if its first non-whitespace character is a `#`. The first name is
    ignored. If the first line of the file does not begin with a `#`, the 
    fields are named `field_1`, `field_2`, and so on, where `field_1` is 
    the name of the second column.'''
    
    for row in parse_metadata(open_md_file):
        _x, docid = row[0]
        if docid in comp:
            comp[docid].update(row[1:])

def load_topic_metadata(topic_md_file):
    topic_md_map = {}
    for row in parse_metadata(topic_md_file):
        _x, tid = row[0]
        tid = int(tid)
        topic_md_map[tid] = dict(row[1:])
    return topic_md_map

def load_and_filter_texts(compfile, parser_rex, metadata, filters):
    texts = parse_composition_file(compfile, parser_rex)
    if metadata is not None:
        add_text_metadata(texts, metadata)
    if filters is not None:
        texts = filter_texts(texts, filters)
    return texts

def shared_topic_top_texts(texts, topics, n):
    ordered = sorted(texts)
    vectors = [[texts[tx][tp] for tx in ordered] for tp in topics]
    multiplied = [reduce(operator.mul, col) for col in zip(*vectors)]
    proportions_texts = sorted(zip(multiplied, ordered), reverse=True)[0:n]
    return proportions_texts

def shared_topic_controller(args):
    texts = load_and_filter_texts(args.composition_file,
                                  args.parser_rex,
                                  args.document_metadata,
                                  args.metadata_filter)
    
    if args.topic_metadata is not None:
        topic_md = load_topic_metadata(args.topic_metadata)

    if args.each_topic:
        for tn in args.topic_num:
            top = shared_topic_top_texts(texts, [tn], args.num_texts)
            print
            print "Topic {}".format(tn),
            if tn in topic_md and 'name' in topic_md[tn]:
                print ": {}".format(topic_md[tn]['name'])
            else:
                print

            shared_topic_view(texts, top, args.metadata_field)
    else:
        msg = "Top texts for topics {}:"
        print
        print msg.format(', '.join(map(str, args.topic_num)))
        if all(tn in topic_md for tn in args.topic_num):
            msg = '\t{}.'
            topic_names = (topic_md[tn]['name'] for tn in args.topic_num)
            print msg.format(', '.join(topic_names))
            print
        top = shared_topic_top_texts(texts, args.topic_num, args.num_texts)
        shared_topic_view(texts, top, args.metadata_field)

def shared_topic_view(texts, top, fields=None):
    fields = [] if fields is None else fields
    for proportion, text in top:
        print proportion, 
        for f in fields:
            default = '[... {} not found ...]'.format(f)
            print ' | ', texts[text].get(f, default),
        print

def construct_doc_topic_matrix(texts):
    maxtopic = 0
    random_t = next(texts.itervalues())
    while maxtopic in random_t:
        maxtopic += 1
    
    texts = [texts[tx] for tx in sorted(texts)]
    mat_rows = [[tx[topic] for topic in xrange(maxtopic)] for tx in texts]
    return numpy.array(mat_rows)

def is_symmetric(mat, eps=2 ** -12):
    return (numpy.abs(mat - mat.T) < eps).all()

def create_text_filter(filters):
    def text_filter(item):
        for key, value in filters:
            if (   key not in item
                or str(item[key]) != value):
                return False
        return True
    return text_filter

def filter_texts(texts, filters):
    text_filter = create_text_filter(filters)
    return {key:texts[key] for key in texts if text_filter(texts[key])}

def topic_graph_add_metadata(graph, topic_md):
    for tid in topic_md:
        if tid < len(graph.node) and tid >= 0:
            graph.node[tid].update(topic_md[tid])

def topic_graph_nx_convert(sim_matrix):
    if is_symmetric(sim_matrix):
        graph = networkx.Graph
    else:
        graph = networkx.DiGraph
    return graph

def topic_graph_controller(args):
    texts = load_and_filter_texts(args.composition_file,
                                  args.parser_rex,
                                  args.document_metadata,
                                  args.metadata_filter)
    
    # Construct similarity matrix
    DTM = construct_doc_topic_matrix(texts)
    sim_func = similarity_dispatcher[args.similarity_function]
    sim_matrix = sim_func(DTM.T, DTM)
    
    if args.remove_self_loops:
        sim_matrix = remove_self_loops(sim_matrix)

    # Construct networkx graph
    if is_symmetric(sim_matrix):
        graph_cons = networkx.Graph
    else:
        graph_cons = networkx.DiGraph
    thresh_func = threshold_dispatcher[args.threshold_function]
    sim_matrix = thresh_func(args.threshold_value, sim_matrix)
    
    # Note: it's typical as far as I can tell to represent a markov chain
    # using a matrix with column vectors that add to one. These represent
    # transition probabilities _to_ row index _from_ given column index. But
    # networkx uses the convention that each directed edge moves _from_ its
    # row index _to_ its column index. So here we transpose. 
    graph = graph_cons(sim_matrix.T)
    
    if args.topic_metadata is not None:
        topic_md = load_topic_metadata(args.topic_metadata)
    else:
        topic_md = []
    
    topic_graph_add_metadata(graph, topic_md)

    if args.calculate_centrality:
        centrality = eigenvector_centrality(sim_matrix)
        topic_graph_view_centrality(graph, centrality, ['name'])
        
        # Sanity check against nx.eigenvector_centrality.
        # This seems to fail when there are unlinked nodes; networkx's power
        # iteration routine fails to converge. That could mean that the result
        # is ill-founded in those cases. But _my_ power iteration code has no 
        # issue with it, so I'm a little confused. I'll have to look into this
        # more.
        #nx_centrality = networkx.eigenvector_centrality(graph)
        #nx_centrality = numpy.array([val for key, val in sorted(nx_centrality.items())])
        #nx_centrality /= nx_centrality.sum()
        #assert (numpy.abs(nx_centrality - centrality) < 2 ** -12).all()

    if args.write_markov_cluster_file is not None:
        mcluster = markov_cluster(
            sim_matrix, 
            power=args.markov_cluster_power, 
            inflate=args.markov_cluster_inflation, 
            selfloop=args.markov_cluster_selfloop)
        mcluster = mcluster.T
        mcluster = networkx.DiGraph(mcluster)
        topic_graph_add_metadata(mcluster, topic_md)
        
        filename = args.write_markov_cluster_file
        topic_graph_save(mcluster, filename, args.output_type)

    if args.write_network_file is not None:
        filename = args.write_network_file
        topic_graph_save(graph, filename, args.output_type)
 
def topic_graph_save(graph, filename, filetype):
    n_edges = len(graph.edges())
    n_nodes = len(graph.nodes())
    directed = isinstance(graph, networkx.DiGraph)
    directed = "Directed" if directed else "Undirected"
    print "Writing {} Graph: {} nodes, {} edges".format(directed, 
                                                        n_nodes,
                                                        n_edges)
    if filetype == 'gexf':
        if not filename.endswith('.gexf'):
            filename += '.gexf'
        networkx.write_gexf(graph, filename)
    elif filetype == 'json':
        if not filename.endswith('.json'):
            filename += '.json'
        data = networkx.readwrite.json_graph.node_link_data(graph)
        with open(filename, 'w') as gfile:
            json.dump(data, gfile)

def topic_graph_view_centrality(graph, centrality, fields):
    if fields is None:
        fields = []
    centrality = ((cval, i) for i, cval in enumerate(centrality))
    centrality = sorted(centrality, reverse=True)
    for rank, (val, topic) in enumerate(centrality):
        msg = '{:4} : {:4}' + '  {}' * (len(fields) + 1)
        data = [rank, topic]
        graph_node_topic = graph.node[topic]
        data.extend([graph_node_topic[field] 
                        if field in graph_node_topic else 
                     ''
                        for field in fields])
        data.append(val)
        print msg.format(*data)

def cosine_similarity(A, B):
    '''A normed dot product, vectorized. If A has n rows, B should
       have n columns. A and B can be vectors or matrices; if one
       or both is a matrix, then the result will be a matrix. For
       higher dimensions, this is undefined.'''
 
    A = numpy.asarray(A)
    B = numpy.asarray(B)
    if A.ndim == 1 and B.ndim == 1:
        norm = numpy.sqrt(numpy.dot(A, A) * numpy.dot(B, B))
        return numpy.dot(A, B) / norm
    elif A.ndim == 1:
        A = A[None,:]
    elif B.ndim == 1:
        B = B[:,None]
    elif A.ndim > 2 or B.ndim > 2:
        raise ValueError('cosine_similarity is undefined for 3-dimensional '
                         'arrays and higher')

    norm = numpy.sqrt(numpy.outer((A * A).sum(axis=1), (B * B).sum(axis=0)))
    return numpy.dot(A, B) / norm

    # The above function uses some slightly dense linear algebra that
    # is worth unpacking here. The above equation for `norm` does the 
    # same thing as `norm` in the equation for the 1-d case. But it
    # does it in a vectorized way; the in-place multiplications and
    # summations are like 1-d dot products applied to each row or
    # column in a matrix, resulting in a 1-d array. Then the outer
    # product multiplies the results in a way that distributes them
    # into the right place. 

    # It's a little hard to explain better than that, but I'll try.
    # Note how matrix multiplication computes a dot product for every
    # combination of row from A and column from B. So it computes a
    # a dot product between row 0 of A and column 1 of B. The
    # denominator here squares row 0 of A and sums it (along with 
    # all other rows of A) and it squares column 1 of B and sums
    # it, etc. Then the outer product multiplies those two values, 
    # and places the result in a matrix, in the exact same position 
    # as the result of the dot product.

    # And in case you were wondering, the outer product is just what
    # you get when you multiply a column vector (on the left) by a
    # row vector (on the right). So in other words, the below formula
    # could replace the outer product with classic numpy broadcasting.

    # norm numpy.sqrt((A * A).sum(axis=1)[:,None] * \
    #                 (B * B).sum(axis=0)[None,:])

    # Note also that if the values passed to this are mean-centered
    # by column, then this gives a pearson correlation matrix, and
    # if you mean-center these but only divide them by the number
    # of rows instead of a norm, you get a covariance matrix. 
    
    # For more such equivalences, see http://tinyurl.com/pqshzyn
    # The observation that so many of these values are just 
    # differently-normed dot products motivated my derivation of
    # the `browsing_similarity` below. 

def cosine_similarity_normed(A, B):
    cs = cosine_similarity(A, B)
    norm = cs.sum(axis=0)[None:]
    return cs / norm

def browsing_similarity(A, B):
    '''Similar to cosine similarity, but with a different (asymmetrical)
       normalization scheme. The resulting matrix represents the probability
       distribution of one sample from A, given a previously fixed
       value from B. Concretely, this is the probability that, given
       that you sampled a word from topic X from one of the texts 
       represented in B, you will now sample a word from topic Y in A.
       In other words, this is the corpus-wide set of probabilities
       p(Y|X) for each topic pairing Y and X, given a single word
       sample taken from an arbitrary text chosen under a uniform
       distribution.'''

    A = numpy.asarray(A)
    B = numpy.asarray(B)
    if A.ndim == 1 and B.ndim == 1:
        return numpy.dot(A, B) / B.sum()
    elif A.ndim == 1:
        A = A[None,:]
    elif B.ndim == 1:
        B = B[:,None]
    elif A.ndim > 2 or B.ndim > 2:
        raise ValueError('browsing_similarity is undefined for 3-dimensional '
                         'arrays and higher')
    
    norm = B.sum(axis=0)[None,:]
    return numpy.dot(A, B) / norm

similarity_dispatcher = { 'cosine'        : cosine_similarity,
                          'browsing'      : browsing_similarity,
                          'cosine-normed' : cosine_similarity_normed,
                          'default'       : browsing_similarity }

def remove_self_loops(sim_matrix):
    sim_matrix = sim_matrix.copy()
    diag = numpy.diag_indices_from(sim_matrix)
    sim_matrix[diag] = 0
    return sim_matrix

def flat_threshold(threshold, sim_matrix):
    sim_matrix = sim_matrix.copy()
    sim_matrix[sim_matrix < threshold] = 0
    return sim_matrix

def underwood_threshold(threshold_increment, sim_matrix):
    symmetric = is_symmetric(sim_matrix)
    
    i = 1
    selection = slice(None, None, None)
    while not (sim_matrix[selection] == 0).all():
        selection = rank_select(i, sim_matrix)
        threshold = threshold_increment * i
        selection_ix = sim_matrix[selection] < threshold
        R, C = selection
        sim_matrix[R[selection_ix], C[selection_ix]] = 0
        i += 1

    if symmetric:
        sim_matrix = numpy.maximum(sim_matrix, sim_matrix.T)
    
    return sim_matrix

def rank_select(threshold_rank, sim_matrix):
    indices = xrange(sim_matrix.shape[1])
    R = []
    C = []
    for col in indices:
        ranked = sim_matrix[:, col].argsort()
        selected = ranked[:-threshold_rank]
        R.extend(selected)
        C.extend([col] * len(selected))
    return (numpy.array(R), numpy.array(C))

def rank_threshold(threshold_rank, sim_matrix):
    if threshold_rank < 1: 
        threshold_rank = 1

    symmetric = is_symmetric(sim_matrix)
    to_cut = rank_select(threshold_rank, sim_matrix)
    sim_matrix[to_cut] = 0

    if symmetric:
        sim_matrix = numpy.maximum(sim_matrix, sim_matrix.T)
    
    return sim_matrix

threshold_dispatcher = { 'rank'      : rank_threshold,
                         'underwood' : underwood_threshold,
                         'flat'      : flat_threshold,
                         'default'   : flat_threshold }

def eigenvector_centrality(sim_matrix):
    pi_eig = numpy.ones((sim_matrix.shape[0], 1))
    pi_eig /= numpy.sqrt((pi_eig * pi_eig).sum())
    old_eig = -pi_eig
    cosine_sim_eig = 0
    while cosine_sim_eig != 1.0:
        old_eig = pi_eig
        pi_eig = numpy.dot(sim_matrix, pi_eig)
        pi_eig /= pi_eig.sum()
        cosine_sim_eig = cosine_similarity(pi_eig.T, old_eig)

    return pi_eig.ravel()

def markov_cluster(sim_matrix, power=2, inflate=2, selfloop=0, eps=0):
    sim_matrix = sim_matrix.copy()

    # modulate self-loops:
    di = numpy.diag_indices(sim_matrix.shape[0])
    if (sim_matrix[di] == 0).all():
        sim_matrix[di] = selfloop
    else:
        sim_matrix[di] = sim_matrix[di] * selfloop

    col_norm = sim_matrix.sum(axis=0)[None,:]
    col_norm[col_norm == 0] = 1  # avoids zero-divide but assumes all positive
    sim_matrix /= col_norm
    
    curr = sim_matrix.copy() + 1
    while (numpy.abs(sim_matrix - curr) > eps).any():
        curr[:] = sim_matrix
        sim_matrix = numpy.linalg.matrix_power(sim_matrix, power)
        sim_matrix **= inflate
        col_norm = sim_matrix.sum(axis=0)[None,:]
        col_norm[col_norm == 0] = 1 # assumes all values positive
        sim_matrix /= col_norm
    return sim_matrix

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A collection of statistical '
        'analysis and visualization scripts for exploring the output of '
        'MALLET\'s topic modeling routines. (Currently only supports vanilla '
        'unigram LDA output.)')
    
    parent_parser = argparse.ArgumentParser(description='Parse a MALLET '
        'composition file.')
    parent_parser.add_argument('-m', '--document-metadata', 
        metavar='filename', type=FileType('r'),
        help='A file containing document metadata in a tab-delimited table, '
        'with the relevant document id in the first column. If '
        'the first row starts with a `#` symbol, its entries will be treated '
        'as column names.')
    parent_parser.add_argument('-M', '--topic-metadata', metavar='filename',
        type=FileType('r'), help='A file containing topic metadata in a '
        'tab-delimited table, with the relevant topic id in the first '
        'column. If the first row starts with a `#` symbol, its entries '
        'will be treated as column names.')
    parent_parser.add_argument('-f', '--metadata-field', type=str, 
        action='append', metavar='field-name', help='A metadata field to '
        'include in the output. May be used multiple times.')
    parent_parser.add_argument('-F', '--metadata-filter', type=str,
        action='append', nargs=2, metavar='field-name/value', 
        help='A metadata field to use for filtering input. Only items with '
        'fields set to the given values will be included.')
    parent_parser.add_argument('-r', '--parser-rex', type=str,
        metavar='regular-expression', help='A regular expression for '
        'extracting metadata from filenames as listed in the MALLET '
        'composition file. Only named groups will be captured. For example, '
        'the named group `(?P<year>\d\d\d\d)` will capture a four-digit '
        'sequence and associate it with the key `year`. To use a customized '
        'file id in the first column of your metadata file, create a group '
        'named `docid`.')

    parent_parser.add_argument('composition_file', type=FileType('r'),
        help='A composition file produced by MALLET (via the '
        '--output-doc-topics option).')

    subparsers = parser.add_subparsers(title='available commands',
        description='For more help, the -h/--help option for each command.')
    
    sim_parser = subparsers.add_parser('shared', 
        parents=[parent_parser], conflict_handler='resolve')
    sim_parser.add_argument('-n', '--num-texts', type=int, default=20, 
        metavar='number', help='The number of texts to display.')
    sim_parser.add_argument('topic_num', type=int, nargs='+', help='A list of '
        'topics to compare, specified by the topic number assigned by '
        'MALLET.')
    sim_parser.add_argument('-e', '--each-topic', action='store_true',
        default=False, help='Rather than printing out texts shared between '
        'the given topics, print out the top texts for each given topic '
        'in sequence.')
    sim_parser.set_defaults(func=shared_topic_controller)

    graph_parser = subparsers.add_parser('network', parents=[parent_parser],
        conflict_handler='resolve', help='Represent the given topics as a '
        'network and save to a file of the given format. Link strengths '
        'and directions are calculated using the given similarity metric '
        'over vectors of document proportions for each topic.')
    #graph_parser.add_argument('-k', '--topic-key-file', metavar='filename',
    #    type=FileType('r'), help='A topic key file produced by '
    #    'MALLET (via the --output-topic-keys option).')
    graph_parser.add_argument('-w', '--write-network-file', type=str,
        metavar='filename', help='Write topic network data to the given '
        'filename.')
    graph_parser.add_argument('-o', '--output-type', choices=['gexf', 'json'],
        default='json', type=str, help='The file format to use when saving '
        'topic network data. Defaults to `json`. This value is ignored if '
        'the --write-network-file option is not selected.')

    graph_parser.add_argument('-s', '--similarity-function',
        choices=similarity_dispatcher.keys(), type=str,
        default='default', help='The similarity function '
        'to use. The `cosine` option uses standard cosine similarity; the '
        '`browsing` option uses a modified cosine similarity formula: '
        '`dot(a, b) / norm` where `norm` is not the product of the euclidean '
        'norms of `a` and `b`, but is instead the manhattan norm of `b`. '
        'The result of this last option is a matrix of topic transition '
        'probabilities that corresponds to a reversible markov chain. The '
        'default value is `browsing`.')
    
    graph_parser.add_argument('-t', '--threshold-function', 
        choices=threshold_dispatcher.keys(), type=str, 
        default='default', help='The threshold function '
        'to use for link-cutting. The `flat` option uses a simple value '
        'threshold. Any link in the network with weight below this value '
        'will be cut. The `underwood` option uses a link-cutting heuristic '
        'developed by Ted Underwood for creating easy-to-read networks '
        'that show only the strongest relationsips between topics. It '
        'combines rank- and value-based thresholding; the top-ranked link '
        'is always included; the next-ranked link is included if it is '
        'above the given threshold value `t`; the next link is included if '
        'it is above `t * 2`; the next, if it is above `t * 3`; and so on. '
        'The `rank` option uses a rank threshold; all links with rank below '
        'this value will be cut, and every node will have precisely this '
        'many links. Ranks start at 1, so the highest-ranked link is given '
        'rank 1, the next-highest-ranked link is given rank 2, and so on. '
        'The default value is `flat`.')
    graph_parser.add_argument('-v', '--threshold-value', type=float, 
        default=0.0, metavar='number', help='The threshold to set for the '
        'selected `--threshold-function`. A threshold of 0 preserves all '
        'links, which is the default behavior.')

    graph_parser.add_argument('-W', '--write-markov-cluster-file', type=str,
        metavar='filename', help='Create a clustered version of the graph '
        'using the Markov Cluster Algorithm and write it to the given '
        'filename.')
    graph_parser.add_argument('-p', '--markov-cluster-power', type=int,
        metavar='integer', default=2, help='The `power` parameter to use for '
        'markov clustering. Larger values cause larger clusters to form.')
    graph_parser.add_argument('-i', '--markov-cluster-inflation', type=float,
        metavar='number', default=2.0, help='The `inflation` parameter to use '
        'for markov clustering. Larger values cause smaller clusters to form.')
    graph_parser.add_argument('-L', '--markov-cluster-selfloop', type=float,
        metavar='number', default=0, help='The `selfloop` parameter to use '
        'for markov clustering. Should be a value between 0.0 and 1.0. '
        'higher values cause nodes to avoid joining clusters if possible. '
        'If the `--remove-self-loops` option is selected, this value is '
        'used as the weight for all self-loops; otherwise, the existing '
        'self-loops are multiplied by this value.')
    graph_parser.add_argument('-c', '--calculate-centrality',
        action='store_true', default=False, help='If this option is '
        'used, the topics will be displayed ordered by their '
        'eigenvector centrality in the network.')
    graph_parser.add_argument('-l', '--remove-self-loops',
        action='store_true', default=False, help='If this option is '
        'used, self-loops will be removed. This may have subtle effects '
        'on centrality measurements, especially for networks based on '
        '`browsing` similarity, which for theoretical consistency assumes '
        'a non-zero probability of transitioning from a topic to itself.')
    graph_parser.set_defaults(func=topic_graph_controller)

    args = parser.parse_args()
    args.func(args)
    
       


