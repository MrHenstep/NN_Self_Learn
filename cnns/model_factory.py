from .configs import ModelConfig, DataMetadata
from . import CNN_model as cnnmodel
from . import CNN_model_2 as cnnflexi
from . import ResNet_model as rn


def build_model(model_cfg: ModelConfig, data_meta: DataMetadata):
    name = model_cfg.model_name.lower()

    if name == "simplecnn":
        model = cnnmodel.SimpleCNN(input_size=data_meta.input_size, num_classes=data_meta.num_classes)
    elif name == "cnnflexi":
        model = cnnflexi.SimpleCNNFlexi(
            input_channels=data_meta.input_channels,
            input_size=data_meta.input_size,
            num_classes=data_meta.num_classes,
        )
        if model_cfg.flexi_arch.lower() == "vgg":
            model.make_VGG()
    elif name == "resnet":
        if data_meta.input_channels != 3:
            raise ValueError("ResNet expects 3-channel inputs; choose a different model for this dataset.")
        imagenet_family = data_meta.dataset_key in {"imagenet", "oxford_pets", "tiny_imagenet"}
        use_projection = model_cfg.use_projection
        if use_projection is None:
            use_projection = imagenet_family
        if imagenet_family:
            model = rn.ResNetIN(
                n_classes=data_meta.num_classes,
                use_projection=use_projection,
                use_residual=model_cfg.use_residual,
            )
        else:
            model = rn.ResNetCF(
                n_classes=data_meta.num_classes,
                resnet_n=model_cfg.resnet_n,
                use_projection=use_projection,
                use_residual=model_cfg.use_residual,
            )
    else:
        raise ValueError(f"Unknown model_name '{model_cfg.model_name}'.")

    return model
