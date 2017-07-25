"""
@author: Noel
@date: 07/07/2017
Class utilising the NetworkX graph object with manual implementation of key algorithms
"""
import networkx as nx
import itertools


class NetworkGraph:

    def __init__(self, graph_input):
        """ Initialise the graph object using the edgelist provided """
        if isinstance(graph_input, list):
            self.graph = nx.DiGraph()
            self.graph.add_weighted_edges_from(graph_input)
        elif isinstance(graph_input, nx.DiGraph):
            self.graph = graph_input

        self.cycles = []            # List of cycles
        self.components = []        # List of connected components
        self.sources = [node for node, indegree in self.graph.in_degree(self.graph.nodes()).items() if indegree == 0]
        self.sinks = [node for node, outdegree in self.graph.out_degree(self.graph.nodes()).items() if outdegree == 0]
        self.maximal_cliques = []
        self.black_holes = []
        self.volcanoes = []
        self.distances = nx.shortest_path_length(self.graph, source=None, target=None, weight='weight')
        self.closures = {}          # Stores the closure set of each node


    def findCycles(self):
        """ Return a list of cycles in the graph """
        for s in self.sources:
            self.recursiveCycle(s, 0, [])


    def recursiveCycle(self, node, count, path):
        """ Recursively apply path search to identify cycles in the graph """
        if node in path:
            # The node has already been visited => cycle
            # Need to remove the path up to the node
            ind = path.index(node)
            p = path[ind:]

            # In the event that this is one node, don't add it
            if len(p) > 1:
                self.cycles.append(path[ind:])

        else:
            # Continue searching descendents
            count += 1
            path.append(node)
            for child in self.graph.successors(node):
                self.recursiveCycle(child, count, path[:])


    def slowCycles(self):
        """ Slow implemtation of cycle finder using path search """
        branch_points = []
        branch_dict = {}

        for s in self.sources:
            path = [s]

            while True:
                last = path[-1]

                out_degree = self.graph.out_degree(last)

                if out_degree == 0:
                    # node is a sink, try previous branch nodes
                    if branch_points:
                        last_branch = branch_points[-1]
                        last = branch_dict[last_branch].pop()

                        if len(branch_dict[last_branch]) == 0:
                            branch_points.remove(last_branch)
                            del (branch_dict[last_branch])
                        ind = path.index(last_branch)
                        path = path[:ind + 1]
                        path.append(last)
                    else:
                        # no remaining branch points => end search
                        break

                else:
                    next_node = self.graph.successors(last)[0]

                    if next_node in path:
                        # A cycle has been found
                        ind = path.index(next_node)
                        cycle = path[ind:]
                        # Ensure cycle hasn't already been found
                        if set(cycle) not in [set(x) for x in self.cycles]:
                            self.cycles.append(cycle)

                        # Need to stop searching and go back to last branch to continue
                        if branch_points:
                            last_branch = branch_points[-1]
                            last = branch_dict[last_branch].pop()

                            if len(branch_dict[last_branch]) == 0:
                                branch_points.remove(last_branch)
                                del (branch_dict[last_branch])
                            ind = path.index(last_branch)
                            path = path[:ind+1]
                            path.append(last)
                            continue
                        else:
                            # no remaining branch points => end search
                            break

                    if out_degree > 1:
                        if last not in branch_points:
                            branch_points.append(last)
                            branch_dict[last] = list(self.graph.successors(last))
                            branch_dict[last].remove(next_node)

                    path.append(next_node)


    def printCycles(self, output_file=None):

        if output_file:
            f = open(output_file, "w")
            for c in self.cycles:
                f.write(','.join(c))
            f.close()

        else:
            for c in self.cycles:
                c = c[:]    # Copy to avoid updating original list object
                c.append(c[0])  # To illustrate returning to starting point
                s = '->'.join([str(x) for x in c])
                print s


    def findVolcanos(self, max_size=50):
        """ Find volcano patterns by flipping direction of all nodes and searching for black holes """

        # First reverse the graph
        self.graph.reverse(copy=False)

        # Find blackholes
        self.volcanoes = self.iBlackHoles(max_size)[:]

        # Return graph to original state
        self.graph.reverse(copy=False)


    def findBlackHoles(self, max_size=50):
        """ Use the blackHole algorithm to find black holes """

        self.black_holes = self.iBlackHoles(max_size)[:]


    def adjacencyMatrix(self):
        """ Return an adjacency matrix repn for the graph """
        M = None
        return M


    def iBlackHoles(self, n):
        """
        Implements the iBlackHole algorithm to find all blackhole patterns in the graph
        See (Li, Xiong, Liu et al.), Proceedings IEEE International Conference on Data Mining,
        ICDM, 2010, (294-303).

        The algorithm prunes incompatible nodes to reduce the search space

        :return: List of blackholes in the graph
        """
        blackholes = []

        for i in range(n, 1, -1):
            P = [node for node, outdegree in self.graph.out_degree(self.graph.nodes()).items() if outdegree < i]
            for v in P:
                # Check if v has a successor not in P. If it does, remove v and all predecessors
                missing_successors = [c for c in self.graph.successors(v) if c not in P]
                if len(missing_successors) > 0:
                    P.remove(v)
                    for p in self.graph.predecessors(v):
                        if p in P:
                            P.remove(p)
            for v in P:
                # Check the closure of v -> those with a size of i are candidate black holes
                v_plus = self.closure(v)
                if len(v_plus) > i:
                    P.remove(v)
                    for p in self.graph.predecessors(v):
                        if p in P:
                            P.remove(p)
                elif len(v_plus) == i:
                    # Add to blackhole list if a superset is not already present
                    subset_indicator = 0
                    for s in blackholes:
                        if set(v_plus).issubset(set(s)):
                            subset_indicator = 1
                            break
                    if subset_indicator == 0:
                        blackholes.append(v_plus)
                    P.remove(v)
                    for p in self.graph.predecessors(v):
                        if p in P:
                            P.remove(p)

            # For the remaining nodes we do a brute force check on all possible subgraphs of size i
            for B in list(itertools.combinations(P, i)):
                B = self.graph.subgraph(B)

                # Check if B is connected
                if nx.is_connected(B.to_undirected()):
                    S = [item for sublist in [self.graph.successors(v) for v in B] for item in sublist]
                    # If B is a blackhole then the union of successors(v) for all v in B is contained in B
                    O = [v for v in S if v not in B.nodes()]
                    if len(O) == 0:
                        subset_indicator = 0
                        for s in blackholes:
                            if set(B).issubset(set(s)):
                                subset_indicator = 1
                                break
                        if subset_indicator == 0:
                            blackholes.append(list(B))

        return blackholes


    def bruteForceBlackholes(self, n):
        """ Find blackholes of size n by checking all possible subgraphs """
        blackholes = []

        for B in list(itertools.combinations(self.graph.nodes(), n)):
            B = self.graph.subgraph(B)

            # Check if B is connected
            if nx.is_connected(B.to_undirected()):
                S = [item for sublist in [self.graph.successors(v) for v in B] for item in sublist]
                # If B is a blackhole then the union of successors(v) for all v in B is contained in B
                O = [v for v in S if v not in B.nodes()]
                if len(O) == 0:
                    blackholes.append(list(B))

        return blackholes


    def closure(self, node):
        """ Find the closure of a node ie all nodes reachable from it """
        node_plus = [node] + list(nx.descendants(self.graph, node))

        return node_plus


    def findMaximalCliques(self):
        """
        Implements the Bron-Kerbosh algorithm to return maximal cliques:
        https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm

         BronKerbosch1(R, P, X):
         if P and X are both empty:
           report R as a maximal clique
         for each vertex v in P:
           BronKerbosch1(R union {v}, P intersect N(v), X intersect N(v))
           P := P less {v}
           X := X union {v}
        """
        self.BronKerbosh(set(), set(self.graph.nodes()), set(), 0)


    def BronKerbosh(self, R, P, X, depth):
        # Comprehensive testing needed on the use of set objects rather than lists
        if len(P) == 0 and len(X) == 0:
            self.maximal_cliques.append(list(R))
            return
        for v in P:
            nbrs = set(nx.all_neighbors(self.graph, v))
            v = set([v])
            new_R=R.union(v)
            new_P=P.intersection(nbrs)
            new_X=X.intersection(nbrs)
            self.BronKerbosh(new_R, new_P, new_X, depth+1)
            P = P.difference(v)
            X = X.union(v)


if __name__ == "__main__":

    # Simple unit tests

    print ("Cycle tests\n")
    edges = [(1, 2, 1), (2, 3, 1), (3, 4, 1), (3, 5, 1), (5, 4, 1), (4, 2, 1)]
    n1 = NetworkGraph(edges)
    n1.slowCycles()
    n1.printCycles()
    print(n1.cycles)

    for n in n1.graph.nodes():
        print list(nx.all_neighbors(n1.graph, n))

    print "\nDistances tests"
    print "d(1, 2) =", n1.distances[1][2]
    print "d(2, 4) =", n1.distances[2][4]
    print "d(1, 5) =", n1.distances[1][5]

    print ("\nClique tests")
    n1.findMaximalCliques()
    print(n1.maximal_cliques)

    print(list(nx.find_cliques(n1.graph.to_undirected())))

    print("\nTest graph input")
    g=nx.DiGraph()
    g.add_weighted_edges_from(edges)

    n2 = NetworkGraph(g)
    print(n2.sources)

    print("\nBlackhole tests")
    edges2 = [(1, 2, 1),
              (2, 3, 1),
              (1, 3, 1),
              (4, 3, 1),
              (5, 1, 1),
              (5, 4, 1),
              (4, 7, 1),
              (4, 6, 1),
              (6, 5, 1),
              (8, 5, 1),
              (5, 9, 1),
              (6, 10, 1),
              (6, 11, 1),
              (9, 12, 1),
              (10, 9, 1),
              (10, 11, 1),
              (11, 12, 1),
              (13, 9, 1),
              (13, 12, 1)]
    n3 = NetworkGraph(edges2)
    n3.findBlackHoles(5)
    n3.findVolcanos(5)

    print(n3.volcanoes)
    print(n3.black_holes)

    print n3.bruteForceBlackholes(3)
    print n3.bruteForceBlackholes(5)




