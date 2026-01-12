import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from lib.geoopt import ManifoldParameter
from lib.geoopt.optim import RiemannianAdam, RiemannianSGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, MultiStepLR

from models.classifier import ResNetClassifier

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # If the original dataset already applied a transform, 
        # we need to work with raw data instead
        return self.transform(img), label

    def __len__(self):
        return len(self.subset)
    

def load_checkpoint(model, optimizer, lr_scheduler, args):
    """ Loads a checkpoint from file-system. """

    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    if 'optimizer' in checkpoint:
        if checkpoint['args'].optimizer == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr

            if (lr_scheduler is not None) and ('lr_scheduler' in checkpoint):
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            print("Warning: Could not load optimizer and lr-scheduler state_dict. Different optimizer in configuration ({}) and checkpoint ({}).".format(args.optimizer, checkpoint['args'].optimizer))

    epoch = 0
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch'] + 1

    return model, optimizer, lr_scheduler, epoch

def load_model_checkpoint(model, checkpoint_path):
    """ Loads a checkpoint from file-system. """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    return model

def select_model(img_dim, num_classes, args):
    """ Selects and sets up an available model and returns it. """

    enc_args = {
        'img_dim' : img_dim,
        'embed_dim' : args.embedding_dim,
        'num_classes' : num_classes,
        'bias' : args.encoder_manifold=="lorentz",
        'init_method' : args.init_method
    }

    if args.encoder_manifold=="lorentz":
        enc_args['learn_k'] = args.learn_k
        enc_args['k'] = args.encoder_k

    dec_args = {
        'embed_dim' : args.embedding_dim,
        'num_classes' : num_classes,
        'k' : args.decoder_k,
        'learn_k' : args.learn_k,
        'type' : 'mlr',
        'clip_r' : args.clip_features
    }

    model = ResNetClassifier(
        num_layers=args.num_layers,
        enc_type=args.encoder_manifold,
        dec_type=args.decoder_manifold,
        enc_kwargs=enc_args,
        dec_kwargs=dec_args,
        
    )

    return model

def select_optimizer(model, args):
    """ Selects and sets up an available optimizer and returns it. """

    model_parameters = get_param_groups(model, args.lr, args.weight_decay)

    if args.optimizer == "RiemannianAdam":
        optimizer = RiemannianAdam(model_parameters, lr=args.lr, weight_decay=args.weight_decay, stabilize=1)
    elif args.optimizer == "RiemannianSGD":
        optimizer = RiemannianSGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, stabilize=1)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model_parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    else:
        raise "Optimizer not found. Wrong optimizer in configuration... -> " + args.model

    lr_scheduler = None
    if args.lr_scheduler == "cosine":
        if args.warmup_epochs > 0:
            warmup_scheduler = LinearLR(optimizer,
                                        start_factor = 0.01,
                                        end_factor=1.0,
                                        total_iters=args.warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max = args.num_epochs - args.warmup_epochs,
                                                 eta_min=0)
            lr_scheduler = SequentialLR(optimizer,
                                        schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[args.warmup_epochs])
        else:
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.num_epochs,
                eta_min=0  # LR goes down to 0
            )
    else:
        if args.warmup_epochs > 0:
            warmup_scheduler = LinearLR(optimizer,
                                        start_factor=0.01,
                                        end_factor=1.0,
                                        total_iters=args.warmup_epochs)
            step_scheduler = MultiStepLR(
                optimizer, milestones=[m - args.warmup_epochs for m in [60, 120, 160]], gamma=0.2
            )
            lr_scheduler = SequentialLR(optimizer,
                                        schedulers=[warmup_scheduler, step_scheduler],
                                        milestones=[args.warmup_epochs])
        else:
            lr_scheduler = MultiStepLR(
                optimizer, milestones=[60, 120, 160], gamma=0.2
            )
        

    return optimizer, lr_scheduler

def get_param_groups(model, lr_manifold, weight_decay_manifold):
    no_decay = ["scale"]
    k_params = ["manifold.k"]

    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_decay)
                and not isinstance(p, ManifoldParameter)
                and not any(nd in n for nd in k_params)
            ],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and isinstance(p, ManifoldParameter)
            ],
            'lr' : lr_manifold,
            "weight_decay": weight_decay_manifold
        },
        {  # k parameters
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and any(nd in n for nd in k_params)
            ], 
            "weight_decay": 0,
            "lr": 1e-4
        }
    ]

    return parameters

def select_dataset(args):
    """ Selects an available dataset and returns PyTorch dataloaders for training, validation and testing. """

    if args.dataset == 'MNIST':
        
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        train_set = datasets.MNIST('data', train=True, download=True, transform=train_transform)
        if args.validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.MNIST('data', train=False, download=True, transform=test_transform)

        img_dim = [1, 32, 32]
        num_classes = 10

    elif args.dataset == 'CIFAR-10':
        train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        train_set_raw = datasets.CIFAR10('data', train=True, download=True, transform=None)
        if args.train_subset_fraction:
            subset_length = int(len(train_set_raw) * args.train_subset_fraction)
            train_set_raw, _ = torch.utils.data.random_split(train_set_raw, [subset_length, len(train_set_raw) - subset_length], generator=torch.Generator().manual_seed(1))

        if args.val_fraction > 0.0:
            new_train_set_length = int(len(train_set_raw) * (1 - args.val_fraction))
            train_subset, val_subset = torch.utils.data.random_split(train_set_raw, [new_train_set_length, len(train_set_raw) - new_train_set_length], generator=torch.Generator().manual_seed(1))
            train_set = TransformSubset(train_subset, train_transform)
            val_set = TransformSubset(val_subset, test_transform)
        else:
            train_set = TransformSubset(train_set_raw, train_transform)
            val_set = datasets.CIFAR10('data', train=False, download=True, transform=test_transform)

        
        test_set = datasets.CIFAR10('data', train=False, download=True, transform=test_transform)
        print(len(train_set), len(val_set), len(test_set))

        img_dim = [3, 32, 32]
        num_classes = 10

    elif args.dataset == 'CIFAR-100':
        train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        train_set_raw = datasets.CIFAR100('data', train=True, download=True, transform=None)
        if args.train_subset_fraction:
            subset_length = int(len(train_set_raw) * args.train_subset_fraction)
            train_set_raw, _ = torch.utils.data.random_split(train_set_raw, [subset_length, len(train_set_raw) - subset_length], generator=torch.Generator().manual_seed(1))

        if args.val_fraction > 0.0:
            new_train_set_length = int(len(train_set_raw) * (1 - args.val_fraction))
            train_subset, val_subset = torch.utils.data.random_split(train_set_raw, [new_train_set_length, len(train_set_raw) - new_train_set_length], generator=torch.Generator().manual_seed(1))
            train_set = TransformSubset(train_subset, train_transform)
            val_set = TransformSubset(val_subset, test_transform)
        else:
            train_set = TransformSubset(train_set_raw, train_transform)
            val_set = datasets.CIFAR100('data', train=False, download=True, transform=test_transform)

        test_set = datasets.CIFAR100('data', train=False, download=True, transform=test_transform)

        print(len(train_set), len(val_set), len(test_set))

        img_dim = [3, 32, 32]
        num_classes = 100

    elif args.dataset == 'Tiny-ImageNet':
        root_dir = "classification/data/tiny-imagenet-200/"
        train_dir = root_dir + "train/images"
        val_dir = root_dir + "val/images"
        test_dir = root_dir + "val/images" # TODO: No labels for test were given, so treat validation as test

        train_transform=transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_set = datasets.ImageFolder(train_dir, train_transform)
        val_set = datasets.ImageFolder(val_dir, test_transform)
        test_set = datasets.ImageFolder(test_dir, test_transform)

        img_dim = [3, 64, 64]
        num_classes = 200

    else:
        raise "Selected dataset '{}' not available.".format(args.dataset)

    # Dataloader
    train_loader = DataLoader(train_set, 
        batch_size=args.batch_size, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=True
    )
    val_loader = DataLoader(val_set, 
        batch_size=args.batch_size_test, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=False
    )
    test_loader = DataLoader(test_set, 
        batch_size=args.batch_size_test, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=False
    ) 
        
    return train_loader, val_loader, test_loader, img_dim, num_classes
