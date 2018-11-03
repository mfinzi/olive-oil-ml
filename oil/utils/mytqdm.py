import inspect
try:    #tqdm.autonotebook
    from tqdm import tqdm_notebook as tqdm
    old_print = print
    # if tqdm.tqdm.write raises error, use builtin print
    def new_print(*args, **kwargs):
        try: tqdm.write(*map(lambda x: str(x),args), **kwargs)
        except: old_print(*args, ** kwargs)
    inspect.builtins.print = new_print
except ImportError: tqdm = lambda it:it