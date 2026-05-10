import torch.optim as optim


def build_training_optimizer(args, model, pipeline_mode):
    if (
        (pipeline_mode == 'frozen' and getattr(args, 'unfreeze_paconv_in_frozen', False))
        or (pipeline_mode == 'e2e' and getattr(args, 'e2e_staged_paconv', False))
    ):
        lr_scale = (
            getattr(args, 'e2e_paconv_lr_scale', 0.1)
            if pipeline_mode == 'e2e'
            else getattr(args, 'frozen_paconv_lr_scale', 1.0)
        )
        if lr_scale <= 0:
            raise ValueError("PAConv lr scale must be positive")

        paconv_params = []
        deeppa_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('stage1_paconv.'):
                paconv_params.append(param)
            else:
                deeppa_params.append(param)

        param_groups = []
        if deeppa_params:
            param_groups.append({'params': deeppa_params, 'lr': args.lr})
        if paconv_params:
            param_groups.append({'params': paconv_params, 'lr': args.lr * lr_scale})

        print(f"[INFO] {pipeline_mode} PAConv optimizer group enabled: PAConv LR scale = {lr_scale}")
        print(f"[INFO] Optimizer param groups: DeepPA={len(deeppa_params)}, PAConv={len(paconv_params)}")
        return optim.Adam(param_groups, lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)

    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
