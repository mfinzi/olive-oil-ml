import argparse
import sys
import numpy as np
from .configGenerator import flatten, unflatten
from oil.datasetup import *
from oil.architectures import *
from torch.optim import *

def argupdated_config(cfg,parser=None, namespace=None):
    """ Uses the cfg to generate a parser spec which parses the command line arguments
        and outputs the updated config. An existing argparser can be specified."""
    # TODO: throw error for clobbered names
    flat_cfg = flatten(cfg)
    if parser is None:
        fmt = lambda prog: argparse.HelpFormatter(prog, max_help_position=80)
        parser = argparse.ArgumentParser(formatter_class=fmt)
        #formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    clobbered_name_mapping = {}
    for field, value in flat_cfg.items():
        fields = field.split('/')
        short_field_name = fields[-1]
        parser.add_argument('--'+short_field_name,default=value,help="(default: %(default)s)")
        clobbered_name_mapping[short_field_name] = field
        if len(fields)>1 and fields[0] not in clobbered_name_mapping:
            parser.add_argument('--'+fields[0],default={},help="Additional Kwargs")
            clobbered_name_mapping[fields[0]] = fields[0]

    parser.add_argument("--local_rank",type=int) # so that distributed will work #TODO: sort this out properly
    args = parser.parse_args()
    if namespace is not None:
        globals().update({k: getattr(namespace, k) for k in namespace.__all__})
    for short_argname, argvalue in vars(args).items():
        if isinstance(argvalue,str):
            try: argvalue = eval(argvalue) # Try to evaluate the strings
            except (NameError, SyntaxError):
                pass # Interpret just as string
        if short_argname in clobbered_name_mapping: # There may be additional args from argparse
            flat_cfg[clobbered_name_mapping[short_argname]] = argvalue
        else:
            flat_cfg[short_argname] = argvalue # Never actually called?

    extra_flat_cfg = flatten(flat_cfg)
    updated_full_cfg = unflatten(extra_flat_cfg) # Flatten again
    return updated_full_cfg
