import numpy as np
import theano as tea
import pydot

def real_simple(x):
    
    a = tea.tensor.scalar() # declare variable
    #b = tea.tensor.dscalar() # declare variable
    g = tea.function([a], a*10);
    
    out = pow(g(a), 2);
    #out = pow(a,2) + pow(b,2) +2*a*b          # build symbolic expression
    f = tea.function([a], out)   # compile function
    
    return g(x) #f(x)