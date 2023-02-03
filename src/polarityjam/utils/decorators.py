import warnings
import functools


def experimental(func):
    """This is a decorator which can be used to mark functions
    as experimental. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', UserWarning)  # turn off filter
        warnings.warn("Call to experimental function {}. Handle results with care!".format(func.__name__),
                      category=UserWarning,
                      stacklevel=2)
        warnings.simplefilter('default', UserWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func
