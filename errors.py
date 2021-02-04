'''
Implementation of custom errors for the family of ThinkComplexity projects
'''

##############################################################
# errors for the pyscf enviro
##############################################################

class PlotError(Exception):
    '''
    Basic error for when things go wrong in the plot module
    '''
    
    pass; #### end PlotError
    
class PlotTypeError(TypeError):
    '''
    When a plot func gets the wrong input type
    '''

    pass; #### end PlotError
    
    

