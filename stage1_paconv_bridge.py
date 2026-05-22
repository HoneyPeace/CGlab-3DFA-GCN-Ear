from types import SimpleNamespace


def is_original_github_stage1(args):
    return getattr(args, "stage1_paconv_source", "default").lower() == "original_github"


def get_stage1_input(args, point_input):
    if is_original_github_stage1(args):
        stage1_in_channels = int(getattr(args, "stage1_in_channels", 3))
        return point_input[:, :stage1_in_channels, :].contiguous()
    return point_input


def build_original_github_stage1(args, landmark_num):
    from legacy_3dfa_gcn.original_paconv_current_split import (
        OriginalPAConv,
        get_assemble_dgcnn,
    )

    original_args = SimpleNamespace(
        k=args.k,
        calc_scores=args.calc_scores,
        hidden=[[16], [16], [16], [16]],
        num_matrices=[8, 8, 8, 8],
        dropout=args.dropout,
        use_cuda_extension=getattr(args, "stage1_use_cuda_extension", False),
        return_latent=True,
        in_channels=int(getattr(args, "stage1_in_channels", 3)),
    )
    original_args.assemble_dgcnn = get_assemble_dgcnn(original_args.use_cuda_extension)
    return OriginalPAConv(original_args, landmark_num)
