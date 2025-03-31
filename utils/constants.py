"""Constants used in the code."""

colors = [
    "#91c5dc",
    "#4692c4",
    "#549236",
    "#ffbb78",
    "#f2a484",
    "#ff7f0e",
    "#ffb347",
    "#ffebd4",
    "#91c5da",
    "#4692c4",
]
line_colors = [
    "#548236",
    "#5A9BD5",
    "#FFC001",
    "#ffbb78",
]

bar_colors = [
    "#91c5dc",
    "#4692c4",
    "#549236",
    "#ffbb78",
]

dataset_name_acronym = {
    "caltech101": "CAL101",
    "cifar100": "CI100",
    "country211": "CO211",
    "cub200": "CUB200",
    "dtd": "DTD",
    "eurosat": "EUSAT",
    "fgvc_aircraft": "AirCr.",
    "food101": "Food101",
    "gtsrb": "GTSRB",
    "mini_imagenet": "MiniIN",
    "oxford_flowers": "FLO102",
    "oxford_pets": "Pets",
    "resisc45": "RES45",
    "stanford_cars": "Cars",
    "sun397": "SUN397",
    "voc2007": "VOC",
}

encoder_name_acronym = {
    "google/vit-base-patch16-224": "ViT-B16-224",
    "google/vit-base-patch16-224-in21k": "ViT-B16-224-21k",
    "google/vit-base-patch16-384": "ViT-B16-384",
    "google/vit-base-patch32-224-in21k": "ViT-B32-224-21k",
    "google/vit-base-patch32-384": "ViT-B32-384",
    "google/vit-large-patch16-224": "ViT-L16-224",
    "google/vit-large-patch16-224-in21k": "ViT-L16-224-21k",
    "google/vit-large-patch16-384": "ViT-L16-384",
    "google/vit-large-patch32-224-in21k": "ViT-L32-224-21k",
    "google/vit-large-patch32-384": "ViT-L32-384",
    "google/vit-huge-patch14-224-in21k": "ViT-H14-224-21k",
}

encoder_params = {
    "google/vit-base-patch16-224": 86.6,
    "google/vit-base-patch16-224-in21k": 86.6,
    "google/vit-base-patch16-384": 86.9,
    "google/vit-base-patch32-224-in21k": 88.0,
    "google/vit-base-patch32-384": 88.3,
    "google/vit-large-patch16-224": 304,
    "google/vit-large-patch16-224-in21k": 304,
    "google/vit-large-patch16-384": 305,
    "google/vit-large-patch32-224-in21k": 306,
    "google/vit-large-patch32-384": 307,
    "google/vit-huge-patch14-224-in21k": 632,
}


dataset_roots = {
    "cub200": "data/CUB_200_2011",
    "mini_imagenet": "data/miniimagenet",
}

fscil_base_classes = {
    "cifar100": 60,
    "cub200": 100,
    "mini_imagenet": 60,
}

fscil_ways = {
    "cifar100": 5,
    "cub200": 10,
    "mini_imagenet": 5,
}

num_classes = {
    "caltech101": 102,
    "cifar100": 100,
    "country211": 211,
    "cub200": 200,
    "dtd": 47,
    "eurosat": 10,
    "fgvc_aircraft": 100,
    "food101": 101,
    "gtsrb": 43,
    "mini_imagenet": 100,
    "oxford_flowers": 102,
    "oxford_pets": 37,
    "resisc45": 45,
    "stanford_cars": 196,
    "sun397": 397,
    "voc2007": 20,
}

fscit_base_classes = {
    "sun397": 45,
    "dtd": 7,
    "voc2007": 2,
    "stanford_cars": 28,
    "resisc45": 5,
    "oxford_pets": 5,
    "oxford_flowers": 12,
    "gtsrb": 7,
    "fgvc_aircraft": 10,
    "eurosat": 1,
    "country211": 22,
    "caltech101": 12,
    "food101": 11,
    "cifar100": 10,
    "cub200": 20,
    "mini_imagenet": 10,
}

fscit_ways = {
    "sun397": 44,
    "dtd": 5,
    "voc2007": 2,
    "stanford_cars": 21,
    "resisc45": 5,
    "oxford_pets": 4,
    "oxford_flowers": 10,
    "gtsrb": 4,
    "fgvc_aircraft": 10,
    "eurosat": 1,
    "country211": 21,
    "caltech101": 10,
    "food101": 10,
    "cifar100": 10,
    "cub200": 20,
    "mini_imagenet": 10,
}

dataset_specific_configs = {
    "google/vit-base-patch16-224-in21k": {
        "FSCIT": {
            "caltech101": {
                "lr_base": 0.1,
                "encoder_ft_start_layer": 6,
                "encoder_ft_start_epoch": 0,
                "encoder_lr_factor": 0.1,
                "incft_layers": "pet",
                "epochs_incremental": 1,
                "inc_ft_lr_factor": 0.01,
            },
            "cifar100": {
                "lr_base": 0.1,
                "encoder_ft_start_layer": 9,
                "encoder_ft_start_epoch": 0,
                "encoder_lr_factor": 0.001,
                "incft_layers": "classifier",
                "epochs_incremental": 1,
                "inc_ft_lr_factor": 0.001,
            },
            "country211": {
                "lr_base": 0.5,
                "encoder_ft_start_layer": 6,
                "encoder_ft_start_epoch": 10,
                "encoder_lr_factor": 0.01,
            },
            "cub200": {
                "lr_base": 0.1,
                "encoder_ft_start_layer": 6,
                "encoder_ft_start_epoch": 25,
                "encoder_lr_factor": 0.1,
                "incft_layers": "classifier+pet",
                "epochs_incremental": 0,
            },
            "dtd": {
                "epochs_base": 30,
                "lr_base": 0.5,
                "encoder_ft_start_layer": -1,
                "encoder_ft_start_epoch": 10,
                "encoder_lr_factor": 0.1,
                "incft_layers": "classifier+pet",
                "epochs_incremental": 1,
                "inc_ft_lr_factor": 0.001,
            },
            "eurosat": {
                "lr_base": 0.5,
                "encoder_ft_start_layer": -1,
                "encoder_ft_start_epoch": 10,
                "encoder_lr_factor": 0.01,
                "incft_layers": "classifier",
                "epochs_incremental": 1,
                "inc_ft_lr_factor": 0.001,
            },
            "fgvc_aircraft": {
                "lr_base": 0.1,
                "encoder_ft_start_layer": 11,
                "encoder_ft_start_epoch": 0,
                "encoder_lr_factor": 0.1,
            },
            "food101": {
                "lr_base": 0.5,
                "encoder_ft_start_layer": 11,
                "encoder_ft_start_epoch": 10,
                "encoder_lr_factor": 1.0,
            },
            "gtsrb": {
                "epochs_base": 30,
                "lr_base": 0.1,
                "encoder_ft_start_layer": 6,
                "encoder_ft_start_epoch": 0,
                "encoder_lr_factor": 0.1,
            },
            "mini_imagenet": {
                "epochs_base": 30,
                "lr_base": 0.1,
                "encoder_ft_start_layer": 3,
                "encoder_ft_start_epoch": 25,
                "encoder_lr_factor": 0.5,
                "incft_layers": "pet",
                "epochs_incremental": 1,
                "inc_ft_lr_factor": 0.001,
            },
            "oxford_flowers": {
                "lr_base": 0.5,
                "encoder_ft_start_layer": -1,
                "encoder_ft_start_epoch": 0,
                "encoder_lr_factor": 0.01,
            },
            "oxford_pets": {
                "lr_base": 0.1,
                "encoder_ft_start_layer": 11,
                "encoder_ft_start_epoch": 25,
                "encoder_lr_factor": 0.5,
                "incft_layers": "classifier+pet",
                "epochs_incremental": 1,
                "inc_ft_lr_factor": 0.001,
            },
            "resisc45": {
                "lr_base": 0.1,
                "encoder_ft_start_layer": 3,
                "encoder_ft_start_epoch": 0,
                "encoder_lr_factor": 0.1,
            },
            "stanford_cars": {
                "epochs_base": 30,
                "lr_base": 0.1,
                "encoder_ft_start_layer": -1,
                "encoder_ft_start_epoch": 0,
                "encoder_lr_factor": 0.01,
            },
            "sun397": {
                "epochs_base": 30,
                "lr_base": 0.1,
                "encoder_ft_start_layer": 11,
                "encoder_ft_start_epoch": 10,
                "encoder_lr_factor": 0.01,
                "incft_layers": "classifier",
                "epochs_incremental": 1,
                "inc_ft_lr_factor": 0.001,
            },
            "voc2007": {
                "epochs_base": 30,
                "lr_base": 0.1,
                "encoder_ft_start_layer": 3,
                "encoder_ft_start_epoch": 25,
                "encoder_lr_factor": 0.1,
                "incft_layers": "classifier",
                "epochs_incremental": 1,
                "inc_ft_lr_factor": 0.001,
            },
        },
        "FSCIL": {
            "cifar100": {
                "lr_base": 0.001,
                "encoder_ft_start_layer": 11,
                "adapt_blocks": 12,
                "encoder_ft_start_epoch": 10,
                "encoder_lr_factor": 0.01,
            },
            "cub200": {
                "lr_base": 0.1,
                "encoder_ft_start_layer": -1,
                "adapt_blocks": 6,
                "encoder_ft_start_epoch": 0,
                "encoder_lr_factor": 0.01,
            },
            "mini_imagenet": {
                "lr_base": 0.001,
                "encoder_ft_start_layer": 11,
                "adapt_blocks": 12,
                "encoder_ft_start_epoch": 10,
                "encoder_lr_factor": 0.01,
            },
        },
    },
}
