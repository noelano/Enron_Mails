import networkx as nx

class NetworkGraph():

    def __init__(self, edgeList):
	    """ Initialise the graph object using the edgelist provided """
		self.g = nx.DiGraph()
        self.g.add_weighted_edges_from(edgeList)
		
    def findCycles(self, outputFile):
	    """ Return a list of cycles in the graph """
	    if outputFile:
		    # Output to csv file
			pass
		
	def findVolcanos(self):
	    pass
		
	def findSinks(self):
	    pass
		
    def findNeighbourhoods(self):
	    pass
