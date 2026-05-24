from torchvision import transforms


def transforms_train_val(augmentation='default', cutout_length=16, randaugment_magnitude=5):
    """Returns train, val, test transforms based on augmentation mode.
    Modes: default, cutout, randaugment, all
    randaugment_magnitude: 3-5 for 84x84 images (9 is for 224x224)
    """
    if augmentation == 'default':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
    elif augmentation == 'cutout':
        from utils.cutout import Cutout
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
            Cutout(n_holes=1, length=cutout_length),
        ])
    elif augmentation == 'randaugment':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=randaugment_magnitude),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
    elif augmentation == 'all':
        from utils.cutout import Cutout
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=randaugment_magnitude),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
            Cutout(n_holes=1, length=cutout_length),
        ])
    else:
        raise KeyError(f'augmentation mode {augmentation} is not supported')

    val_transforms = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    return train_transforms, val_transforms, test_transforms


def transforms_test():
    return transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
