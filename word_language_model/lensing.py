import torch
import torch.nn.functional as F

import fire
import pandas as pd
import numpy as np

class Lens(object):
    def __init__(self, lens, entities):
        """
        Similarity lens.

        Parameters
        ----------
        lens: torch.Tensor. 
              shape (n,n)
        entities: list
              len n
        """
        self._lens = lens
        self._entities = entities

    def expand_lens(self, all_entities):
        expanded_lens = expand_lens(self._lens, self._entities, all_entities)
        return Lens(expanded_lens, all_entities)

    def partition_lens(self, p=0.5, **kwargs):
        """
        p is the proportion of values omitted from the first partition.
        
        - with p=0.3, expect a 70-30 split.
        - with p=0.7, expect a 30-70 split.
        
        omitted values are replaced with NaN
        """
        part1, part2 = partition(self._lens, p=p, **kwargs)
        return Lens(part1, self._entities), Lens(part2, self._entities)

def expand_lens(lens, lensed_entities, all_entities, fill_value=float('NaN')):
    """
    Returns a lens with shape `(|all_entities| , |all_entities|)`
    containing all the values from `lens`, and NaN everywhere else.

    Parameters
    ----------
    - lens: tensor with dims (m, m)
    - lensed_entities: list of m elements
    - all_entities: list of n elements

    Returns:
    --------
    tensor with dims (n,n)
    """
    if lens is None: return None

    is_cuda = lens.is_cuda
    lens = lens.cpu().numpy()

    n = len(all_entities)

    index_all = {elt:i for i, elt in enumerate(all_entities)}
    
    index_map = {} # {lens_i : all_i }
    for i, elt in enumerate(lensed_entities):
        if elt in index_all:
            index_map[i] = index_all[elt] 

    lens_idx, all_idx = zip(*index_map.items())

    expanded_lens = np.full( (n,n), fill_value )
    expanded_lens[np.ix_(all_idx, all_idx)] = lens[np.ix_(lens_idx, lens_idx)]

    expanded_lens = torch.from_numpy(expanded_lens).float()

    if is_cuda:
        expanded_lens = expanded_lens.cuda()
    
    return expanded_lens


def expand_lens_zeros(x, lensed_entities, all_entities):
    """ 
    Returns a lens with shape `(|all_entities| , |all_entities|)`
    containing all the values from `lens`, and 0 everywhere else.

    *** REQUIRES that the lens, x, be stored on cpu. ***

    Parameters
    ----------
    - lens: tensor with dims (m, m)
    - lensed_entities: list of m elements
    - all_entities: list of n elements

    Returns:
    --------
    tensor with dims (n,n)
    
    Adapted from justheuristic's answer at https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809/3
    """
    index_all = {elt:i for i, elt in enumerate(all_entities)}
    
    index_map = {} # {lens_i : all_i }
    for i, elt in enumerate(lensed_entities):
        if elt in index_all:
            index_map[i] = index_all[elt] 

    lens_idx, all_idx = zip(*index_map.items())

    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x) # Zeros don't need to be explicitly stored in sparse form.
        
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    
    relevant_indices = []
    mapped_indices = []
    for coord in indices:
        if int(coord[0]) in index_map and int(coord[1]) in index_map:
            relevant_indices.append( [int(coord[0]), int(coord[1])] )
            mapped_indices.append( [index_map[int(coord[0])], index_map[int(coord[1])]] )
    relevant_indices = torch.LongTensor(relevant_indices)
    relevant_indices = relevant_indices.t()
    values = x[tuple(relevant_indices[i] for i in range(relevant_indices.shape[0]))]

    mapped_indices = torch.LongTensor(mapped_indices)
    mapped_indices = mapped_indices.t()
    
    expanded_size = (len(all_entities), len(all_entities))
    sparse_lens = sparse_tensortype(mapped_indices, values, expanded_size)
    
    return sparse_lens.to_dense()

def load_lens(path, cuda):
    """
    Parameters
    ----------
    path: string
        Path to a header-less csv file or a saved torch tensor containing a lens.
    cuda: bool
        Return the lens as a torch.FloatTensor or a torch.cuda.FloatTensor
   
    Returns
    -------
    torch.cuda.Tensor or torch.Tensor 
    """
    if path.endswith('.csv'):
        lens = pd.read_csv(path, header=None)
        lens = lens.astype('float').as_matrix()
        lens = torch.FloatTensor(lens)
    else:
        lens = torch.load(path)
    return lens.cuda() if cuda else lens

def partition(x, p=0.5, mask_value=float('NaN')):
    """
    Splits a variable x into two partitions. Elements
    in the first partition are replaced by mask_value with
    probability p. Elements left unchanged in the first
    partition are masked in the secnod partition.
    
    """
    mask = x.new(x.size()).bernoulli_(1 - p)                                                                   
    inverse_mask = 1-mask
    mask[mask==0] = mask_value
    inverse_mask[inverse_mask==0] = mask_value
    return (mask * x), (inverse_mask * x)

def load_lens_partitions(train_lens, eval_lens, cuda):
    """
    Parameters
    ----------
    train_lens: str
        Path to a header-less csv file or a saved torch tensor containing a lens.
    eval_lens: str or float or None
        str: Path to a header-less csv file or a saved torch tensor containing a lens.
        float: Proportion of train_lens to be used as eval_lens partition.
        None: No eval_lens.
    cuda: bool
        Return the lens as a torch.FloatTensor or a torch.cuda.FloatTensor
    """
    if train_lens:
        train_lens = load_lens(train_lens, cuda)
        if isinstance(eval_lens, float):
            train_lens, eval_lens = partition(train_lens, p=eval_lens, mask_value=float('NaN'))
        elif eval_lens is not None:
            eval_lens = load_lens(eval_lens, cuda)
    elif eval_lens is not None:
        assert isinstance(eval_lens, str)
        eval_lens = load_lens(eval_lens, cuda)
    return train_lens, eval_lens


def lensed_cross_entropy(input, target, confusion_lens, size_average=True):
    logQ_batch = F.log_softmax(input, 1)
    Cw_batch = confusion_lens[target]
    loss = F.kl_div(logQ_batch, Cw_batch, size_average=False)
    if size_average:
        return loss / input.shape[0]
    else:
        return loss

class LensedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, confusion_lens):
        super(LensedCrossEntropyLoss, self).__init__()
        self.confusion_lens = confusion_lens
    def forward(self, input, target, size_average=True):
        return lensed_cross_entropy(input, target, self.confusion_lens, size_average=size_average)
    
if __name__=='__main__':
  fire.Fire()
