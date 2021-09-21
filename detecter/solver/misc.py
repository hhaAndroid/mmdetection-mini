
__all__=['filter_no_grad_params']

def filter_no_grad_params(model,optimizer_cfg):
    params= []
    memo = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            params += [{"params": [value], "lr": optimizer_cfg['lr'], "weight_decay": optimizer_cfg['weight_decay']}]
    return params
