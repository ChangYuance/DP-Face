from .M3DFEL import M3DFEL
from .M3DFEL_AUs import M3DFEL_AUs

def create_model(args):
    """create model according to args

    Args:
        args
    """
    print(args.fusionmodel)
    if args.fusionmodel == 'M3DFEL_AUs_add':
        from .M3DFEL_AUs_add import M3DFEL_AUs_add
        model = M3DFEL_AUs_add(args)
        return model
    if args.fusionmodel == 'M3DFEL_AUs_add_center_loss':
        from .M3DFEL_AUs_add_center_loss import M3DFEL_AUs_add_center_loss
        model = M3DFEL_AUs_add_center_loss(args)
        return model
    if args.fusionmodel == 'M3DFEL_AUs_FiLM':
        from .M3DFEL_AUs_FiLM import M3DFEL_AUs_FiLM
        model = M3DFEL_AUs_FiLM(args)
    if args.fusionmodel == 'M3DFEL_AUs_FiLM_center_loss':
        from .M3DFEL_AUs_FiLM_center_loss import M3DFEL_AUs_FiLM_center_loss
        model = M3DFEL_AUs_FiLM_center_loss(args)
    if args.fusionmodel == 'M3DFEL_AUs_FiLM2':
        from .M3DFEL_AUs_FiLM2 import M3DFEL_AUs_FiLM2
        model = M3DFEL_AUs_FiLM2(args)
        return model
    if args.fusionmodel == 'M3DFEL_AUs_att':
        from .M3DFEL_AUs_att import M3DFEL_AUs_att
        model = M3DFEL_AUs_att(args)
        return model
    if args.fusionmodel == 'M3DFEL_AUs_att_center_loss':
        from .M3DFEL_AUs_att_center_loss import M3DFEL_AUs_att_center_loss
        model = M3DFEL_AUs_att_center_loss(args)
        return model
    if args.fusionmodel == 'M3DFEL_AUs_only':
        from .M3DFEL_AUs_only import M3DFEL_AUs_only
        model = M3DFEL_AUs_only(args)
        return model
    if args.fusionmodel == 'M3DFEL_AUs_only_center_loss':
        from .M3DFEL_AUs_only_center_loss import M3DFEL_AUs_only_center_loss
        model = M3DFEL_AUs_only_center_loss(args)
        return model
    if args.fusionmodel == 'M3DFEL_dropout':
        from .M3DFEL_dropout import M3DFEL_dropout
        model = M3DFEL_dropout(args)
        return model
    if args.fusionmodel == 'M3DFEL_sample1':
        from .M3DFEL_sample1 import M3DFEL_sample1
        model = M3DFEL_sample1(args)
        return model
    if args.fusionmodel == 'M3DFEL_handmade':
        from .M3DFEL_handmade import M3DFEL_handmade
        model = M3DFEL_handmade(args)
        return model
    if args.fusionmodel == 'M3DFEL_sample1_center_loss':
        from .M3DFEL_sample1_center_loss import M3DFEL_sample1_center_loss
        model = M3DFEL_sample1_center_loss(args)
        return model
    if args.fusionmodel == 'MAE_only_x':
        from .MAE_only_x import MAE_only_x
        model = MAE_only_x(args)
        return model
    if args.fusionmodel == 'MAE_x_AUs':
        from .MAE_x_AUs import MAE_x_AUs
        model = MAE_x_AUs(args)
        return model
    if args.fusionmodel == 'MAE_only_x_s':
        from .MAE_only_x_s import MAE_only_x_s
        model = MAE_only_x_s(args)
        return model
    if args.fusionmodel == 'MAE_only_x_s_center_loss':
        from .MAE_only_x_s_center_loss import MAE_only_x_s_center_loss
        model = MAE_only_x_s_center_loss(args)
        return model
    if args.fusionmodel == 'MAE_x_AUs_s':
        from .MAE_x_AUs_s import MAE_x_AUs_s
        model = MAE_x_AUs_s(args)
        return model
    if args.fusionmodel == 'MAE_x_AUs_s_center_loss':
        from .MAE_x_AUs_s_center_loss import MAE_x_AUs_s_center_loss
        model = MAE_x_AUs_s_center_loss(args)
        return model
    if args.fusionmodel == 'MAE_x_x_a_s':
        from .MAE_x_x_a_s import MAE_x_x_a_s
        model = MAE_x_x_a_s(args)
        return model
    if args.fusionmodel == 'MAE_x_x_a_s_center_loss':
        from .MAE_x_x_a_s_center_loss import MAE_x_x_a_s_center_loss
        model = MAE_x_x_a_s_center_loss(args)
        return model
    if args.fusionmodel == 'M3DFEL_AUs':
        from .M3DFEL_AUs import M3DFEL_AUs
        model = M3DFEL_AUs(args)
        return model
    if args.fusionmodel == 'M3DFEL_AUs_center_loss':
        from .M3DFEL_AUs_center_loss import M3DFEL_AUs_center_loss
        model = M3DFEL_AUs_center_loss(args)
        return model
    if args.fusionmodel == 'M3DFEL':
        from .M3DFEL import M3DFEL
        model = M3DFEL(args)
        return model
    if args.fusionmodel == 'M3DFEL_center_loss':
        from .M3DFEL_center_loss import M3DFEL_center_loss
        model = M3DFEL_center_loss(args)
        return model
    if args.fusionmodel == 'M3DFEL_v2':
        from .M3DFEL_v2 import M3DFEL_v2
        model = M3DFEL_v2(args)
        return model
    if args.fusionmodel == 'Palsynet3d':
        from .Palsynet3d import Palsynet3d
        model = Palsynet3d(args)
        return model
    if args.fusionmodel == 'FP_VGGFace':
        from .FP_VGGFace import FP_VGGFace
        model = FP_VGGFace(args)
        return model

    if args.fusionmodel == 'face_x':
        from .face_x import face_x
        llm_embedding_dim = 512  # 或者你想从 args 里配置
        model = face_x(llm_embedding_dim=llm_embedding_dim)
        return model
    return model
