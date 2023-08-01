def _init():
    global GLOBALS_DICT
    GLOBALS_DICT = {}
 
def set(name, value):
    try:
        GLOBALS_DICT[name] = value
        return True
    except KeyError:
        return False
 
def get(name):
    try:
        return GLOBALS_DICT[name]
    except KeyError:
        return "Not Found"