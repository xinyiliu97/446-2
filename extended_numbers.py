import numpy as np
class C:
    
    def __init__(self, data):
        self.data = np.array(data)
        
    def __repr__(self):
        return "%s + %s i" %(self.data[0], self.data[1])
    def __eq__(self, other):
        return np.allclose(self.data, other.data)
    def __add__(self, other):
        return C(self.data + other.data)