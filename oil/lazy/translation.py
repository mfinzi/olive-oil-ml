import numpy as np
import torch

linalg_trans = {
    'ndarray': [np.ndarray, torch.Tensor],
    'new_zeros': [lambda M,s: np.zeros(s,dtype=M.dtype), lambda M, s: M.new_zeros(*s)],
    'eye': [np.eye, torch.eye],

}
lang_map = {np.ndarray:0,torch.Tensor:1}

@property
def T(self):
    return self if len(self.shape)==1 else self.t()

torch.Tensor.T = T

class Translator(object):
    def __init__(self,translation_set,lang_map):
        self.language = np.ndarray
        self.translation_set = translation_set
        self.lang_map = lang_map
    def set_lang(self,cls):
        if cls is not None: self.language = cls
        self.index = self.lang_map[self.language]
    
    def __getattr__(self,name):
        return self.translation_set[name][self.index]

translated_methods = Translator(linalg_trans, lang_map)

