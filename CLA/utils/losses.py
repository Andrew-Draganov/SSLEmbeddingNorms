import torch
import torch.nn.functional as F

class RescaleNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, power=0):
        ctx.save_for_backward(z)
        ctx.power = power
        return z

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[0]
        power = ctx.power
        norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)

        return grad_output * norm**power, None

def repulsive_loss(z_1, z_2, t=0.5):
    sims = z_1 @ z_2.t() / t

    # Set the diagonals to be large negative values, which become essentially zeros after softmax.
    sims[torch.arange(z_1.size()[0]), torch.arange(z_1.size()[0])] = -1e5

    exp = torch.exp(sims)
    sum_exp = torch.sum(exp, dim=-1)
    log_sum_exp = torch.log(sum_exp)

    return torch.mean(log_sum_exp)

def attractive_loss(z_1, z_2, t=0.5):
    sim = z_1 * z_2
    sim = torch.sum(sim, dim=-1)
    total_loss = torch.mean(sim) / t

    return -total_loss

def nt_xent(z_1, z_2, t=0.5, power=0):
    z_1 = RescaleNorm.apply(z_1, power)
    z_2 = RescaleNorm.apply(z_2, power)

    z_1 = F.normalize(z_1, dim=1)
    z_2 = F.normalize(z_2, dim=1)
    attr_loss = attractive_loss(z_1, z_2, t=t)
    rep_loss = repulsive_loss(z_1, z_2, t=t)
    return attr_loss + rep_loss

def symmetric_attractive_loss(z_1, p_1, z_2, p_2, t=1, power=0):
    p_1 = RescaleNorm.apply(p_1, power)
    p_2 = RescaleNorm.apply(p_2, power)

    z_1 = F.normalize(z_1, dim=1) # the projections have a stop-grad applied to them, so we don't need to rescale their gradients
    p_1 = F.normalize(p_1, dim=1)
    z_2 = F.normalize(z_2, dim=1)
    p_2 = F.normalize(p_2, dim=1)

    loss = attractive_loss(z_1, p_2, t=t)
    loss += attractive_loss(z_2, p_1, t=t)

    return loss / 2

LOSS_FUNC_DICT = {
    'infonce': nt_xent,
    'only_repulsions': repulsive_loss,
    'only_attractions': attractive_loss,
}

