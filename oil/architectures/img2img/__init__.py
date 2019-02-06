#from .densenetFC import FCDenseNet57, FCDenseNet67, FCDenseNet103
from .fcn32s import FCN32s
from .fcn16s import FCN16s
from .fcn8s import FCN8s
from .fcn8s import FCN8sAtOnce
from .vgg import VGG16
#'FCDenseNet57', 'FCDenseNet67', 'FCDenseNet103'
__all__ = ['FCN32s','FCN16s','FCN8s','VGG16']