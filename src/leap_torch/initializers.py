def create_instance(cls, *args, **kwargs):
    """
    Returns a function that instantiates a given class using the
    provided arguments.
    
    param: cls: the class to instantiate.
    param: args: the arguments to be forwarded to the class constructor.
    param: kwargs: the keyward arguments to be forwarded to the class
        constructor.
    returns: a function that when called returns a class instantiated
        using the provided arguments.
    """
    
    def _instantiate(*_, **__):
        return cls(*args, **kwargs)
    return _instantiate