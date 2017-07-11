"""
@author: Noel
@date: 07/07/2017
Class utilising the NetworkX graph object with manual implementation of key algorithms
"""
import networkx as nx
import pandas as pd


class NetworkGraph:

    def __init__(self, input):
        """ Initialise the graph object using the edgelist provided """
        if isinstance(input, list):
            self.graph = nx.DiGraph()
            self.graph.add_weighted_edges_from(input)
        elif isinstance(input, nx.DiGraph):
            self.graph = input

        self.cycles = []            # List of cycles
        self.components = []        # List of connected components
        self.sources = [node for node, indegree in self.graph.in_degree(self.graph.nodes()).items() if indegree == 0]
        self.sinks = [node for node, outdegree in self.graph.out_degree(self.graph.nodes()).items() if outdegree == 0]
        self.maximal_cliques = []

    def findCycles(self, outputFile=None):
        """ Return a list of cycles in the graph """
        for s in self.sources:
            self.recursiveCycle(s, 0, [])

        if outputFile:
            # TODO: Output to csv file
            pass

    def recursiveCycle(self, node, count, path):
        """ Recursively apply path search to identify cycles in the graph """
        if node in path:
            # The node has already been visited => cycle
            # Need to remove the path up to the node
            ind = path.index(node)
            self.cycles.append(path[ind:]+[node])
            # Note that to better display the cycle, the node is added again to bookend it
            # This may need to be dropped for producing the output
        else:
            # Continue searching descendents
            count += 1
            path.append(node)
            for child in self.graph.successors(node):
                self.recursiveCycle(child, count, path[:])

    def printCycles(self):
        for c in self.cycles:
            s = '->'.join([str(x) for x in c])
            print s

    def findVolcanos(self):
        pass

    def findSinks(self):
        pass

    def findCliques(self):
        pass

    def adjacencyMatrix(self):
        """ Return an adjacency matrix repn for the graph """
        M = None
        return M

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

    edges2 = [(1, 2, 1), (2, 3, 1), (3, 4, 1), (3, 5, 1), (5, 4, 1), (4, 2, 1)]
    n2 = NetworkGraph(edges2)
    n2.findCycles()
    n2.printCycles()

    for n in n2.graph.nodes():
        print list(nx.all_neighbors(n2.graph, n))

    n2.findMaximalCliques()
    print(n2.maximal_cliques)

    print(list(nx.find_cliques(n2.graph.to_undirected())))

    print("Test graph input")
    g=nx.DiGraph()
    g.add_weighted_edges_from(edges2)

    n3 = NetworkGraph(g)
    print(n2.sources)


