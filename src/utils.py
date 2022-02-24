_gensym_count = [0]
def gensym():
    """ Get a unique ID. """
    _gensym_count[0] += 1
    return _gensym_count[0]
