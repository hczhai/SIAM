'''
Implementation of custom errors for the family of ThinkComplexity projects
'''

##############################################################
# errors for the graph based projects
##############################################################

class GraphError(Exception):
    '''
    Basic error for when things go wrong when using a Graph object (as defined
    in the GraphObjects.py module).
    '''
    
    pass; #### end GraphError
    
    
class RandomGraphError(GraphError):
    '''
    Error class that inherits from GraphErrors, but is more specific to problems
    with the RandomGraph class defined in the RandomGraph.py module.
    '''
    
    pass; #### end RandomgraphError
    
    
class SmallWorldGraphError(RandomGraphError):
    '''
    Even more specific error for raising when there is a problem with a
    SmallWorldGraph as defined in the SmallWorldGraph.py module.
    '''
    
    pass; #### end SmallWorldGraphError
