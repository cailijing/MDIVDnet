import torch
import time

def denoise_seq(seq, noise, args, model_temporal):
    # add noise
    seqn = noise.addnoise(seq, args, 'val')
    #start time
    seq_time = time.time()
    denframes = torch.clamp(model_temporal(seqn.unsqueeze(dim=0)).mean(dim=2, keepdim=True), 0., 1.).squeeze(dim=0)
    # free memory up
    torch.cuda.empty_cache()
    return seq_time, seqn, denframes


