'''
Functions which are generally helpful in the pyscf enviro
'''

##############################################################
#### constructing atom geometries

def ParseGeoDict(d):
    '''
    Given a dictionary of element names : cartesian coordinates np array,
    turn into a string formatted to pass to the mol.atom attribute
    '''
    
    # iter over elements aka keys
    for el in d:
    
        # get coords
        coords = d[el];
        coords = str(coords)[1:-1]; # get space seperated #s with [s deleted
        print(coords);
    


    
    

