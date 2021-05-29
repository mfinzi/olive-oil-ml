import inspect
try:    #tqdm.autonotebook
    from tqdm.auto import tqdm#_notebook as tqdm
    tqdm.get_lock().locks = []
#     old_print = print
    # if tqdm.tqdm.write raises error, use builtin print
#     def new_print(*args, **kwargs):
#         try: tqdm.write(*map(lambda x: str(x),args), **kwargs)
#         except: old_print(*args, ** kwargs)
#     inspect.builtins.print = new_print
except ImportError: tqdm = lambda it,*args,**kwargs:it
