import torch
from torch.optim import Optimizer
import numpy as np
class CustomOptimizer(Optimizer):
    """Implements Iterative and One Shot Pruning on top of SGD algorithm.

   
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)

    
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, momentum=0, dampening=0,
                 weight_decay=0,nesterov=False,prune_epoch=60,step_of_prune=0,
                 perc_to_prune=0,len_step=0,unfreeze_epoch=100,one_shot_prune=0,
                 iterative_prune=0,epochs_to_finetune=0,epochs_to_densetrain=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,prune_epoch=prune_epoch,
        step_of_prune=step_of_prune,perc_to_prune=perc_to_prune,len_step=len_step,
        unfreeze_epoch=unfreeze_epoch,
        weight_decay=weight_decay, nesterov=nesterov,one_shot_prune=one_shot_prune,iterative_prune=iterative_prune,
        epochs_to_finetune=epochs_to_finetune,epochs_to_densetrain=epochs_to_densetrain)

      

        if iterative_prune != 0 and (epochs_to_finetune == 0 or epochs_to_densetrain == 0):
            raise ValueError("Epochs to fine or Epochs to densetrain can't be zero with iterative pruning")

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CustomOptimizer, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state["mask"] = torch.ones_like(p)
                state["prune"] =-1
                state["unfreeze"]=-1

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
          
            EPS = 1e-6
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                state = self.state[p]
                state['step'] += 1

                mask = state["mask"]    
                prune = state["prune"]
                unfreeze = state["unfreeze"]

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p,alpha = 1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                

                p.data.add_(d_p * mask, alpha = -group['lr'] )

                
               
                
                if group["one_shot_prune"] !=0:

                    if state["step"] == group["prune_epoch"]*group['len_step']+group["step_of_prune"]:
                        q = torch.quantile(abs(p),group["perc_to_prune"])
                        print(q)
                        p[abs(p)<q] = 0
                        p_np = p.cpu().detach().numpy()
                        mask_np = mask.cpu().detach().numpy()
                        mask_np = np.where(abs(p_np) < EPS  , 0 , mask_np)
                        state["mask"] = torch.from_numpy(mask_np).to(torch.device('cuda'))
                       
                       
                        if group["iterative_prune"]!=0:
                            state["prune"] =  group["unfreeze_epoch"] + group["epochs_to_densetrain"]

                            print("Next prune epoch is ",state["prune"])

                        else:
                            print("This is one shot pruning")

                    if group["iterative_prune"]!=0:
                    
                        if state["step"] == group["unfreeze_epoch"]*group['len_step']:

                            state["mask"] = torch.ones_like(p)
                            print("Weights unfreezed")
                            state["unfreeze"] = state["prune"] + group["epochs_to_finetune"]

                            print("Next unfreeze epoch is ",state["unfreeze"])
                            

                        if state["step"] == prune*group['len_step']+group["step_of_prune"]:
                            q = torch.quantile(abs(p),group["perc_to_prune"])
                            print(q)
                            p[abs(p)<q] = 0
                            p_np = p.cpu().detach().numpy()
                            mask_np = mask.cpu().detach().numpy()
                            mask_np = np.where(abs(p_np) < EPS  , 0 , mask_np)
                            state["mask"] = torch.from_numpy(mask_np).to(torch.device('cuda'))

                            state["prune"]= state["unfreeze"] + group["epochs_to_densetrain"]
                            print("Next prune epoch is ",state["prune"])

                        if state["step"] == unfreeze*group['len_step']:

                            state["mask"] = torch.ones_like(p)
                            print("Weights unfreezed")
                            state["unfreeze"] = state["prune"] + group["epochs_to_finetune"]
                            print("Next unfreeze epoch is ",state["unfreeze"])

                
        return loss
