
def create_model(args):
    """create model according to args"""
    model_name = args.fusionmodel
    print(f'fusionmodel: {model_name}')

    # AU+FiLM fusion model
    if model_name == 'M3DFEL_AUs_FiLM_center_loss':
        from .M3DFEL_AUs_FiLM_center_loss import M3DFEL_AUs_FiLM_center_loss
        model = M3DFEL_AUs_FiLM_center_loss(args)
        return model

    # Default: Base model (handles Base, M3DFEL, M3DFEL_center_loss, etc.)
    from .Base import Base
    model = Base(args)
    return model
