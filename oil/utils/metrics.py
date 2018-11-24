from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from scipy.linalg import norm,sqrtm
from ..utils.utils import Eval
#from torch.nn.functional import adaptive_avg_pool2d
#from .pytorch-fid.fid_score import calculate_frechet_distance
#from .pytorch-fid.inception import InceptionV3

# TODO: cache logits for existing datasets
#       should be possible if we can serialize dataloaders

def get_inception():
    """ grabs the pytorch pretrained inception_v3 with resized inputs """
    inception = inception_v3(pretrained=True,transform_input=False)
    model = nn.Sequential(nn.Upsample(size=(299,299),mode='bilinear'),inception)
    return model

def get_logits(model,loader):
    """ Extracts logits from a model, dataloader returns a numpy array of size (N, K)
        where K is the number of classes """
    with Eval(model), torch.no_grad():
        model_logits = lambda mb: model(mb.cuda()).cpu().data.numpy()
        logits = np.concatenate([model_logits(minibatch) for minibatch in loader],axis=0)
    return logits

def FID(loader1,loader2):
    """ Computes the Frechet Inception Distance  (FID) between the two image dataloaders
        using pytorch pretrained inception_v3. Requires >2048 imgs for comparison
        Dataloader should be an iterable of minibatched images, assumed to already
        be normalized with mean 0, std 1 (per color)
        """
    model = get_inception().cuda()
    logits1 = get_logits(model,loader1)
    logits2 = get_logits(model,loader2)
    mu1 = np.mean(logits1,axis=0)
    mu2 = np.mean(logits2,axis=0)
    sigma1 = np.cov(logits1, rowvar=False)
    sigma2 = np.cov(logits2, rowvar=False)

    tr = np.trace(sigma1 + sigma2 - 2*sqrtm(sigma1@sigma2))
    distance = norm(mu1-mu2)**2 + tr
    return distance

def IS(loader):
    """Computes the Inception score of a dataloader using pytorch pretrained inception_v3"""
    model = get_inception().cuda()
    logits = get_logits(model,loader)
    Pyz = np.exp(logits).transpose() # Take softmax (up to a normalization constant)
    Py = Pyz.mean(-1)                # Average over z
    logIS = entropy(Pyz,Py).mean()   # Average over z
    raise np.exp(logIS)