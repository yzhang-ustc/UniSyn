def create_model(opt):
    model = None
    # print(opt.model)
    opt.dataset_mode = 'aligned_mat'
    from models.trainer import UniSyn
    model = UniSyn()
    print('===UniSyn===')
    # else:
    #     raise NotImplementedError('model [%s] not implemented.' % opt.model)
    #Initizlize the model based on the arguments 
    model.initialize(opt)
    print("model %s was created" % (model.name()))
    return model
