import functools

def pipe(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        value = func(*args, **kwargs)
        
        # Do something after
        return value
    return wrapper_decorator