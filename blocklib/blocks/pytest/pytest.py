

class pytest():

    def __init__(self,*args, **kwargs):
        self.available_implementations = [
            'numpy',
            ]
            
        impl = kwargs['impl'] if 'impl' in kwargs else self.available_implementations[0]
        
        if impl == 'numpy':
            from pytest_numpy import pytest_numpy
            return pytest_numpy(*args, **kwargs)

