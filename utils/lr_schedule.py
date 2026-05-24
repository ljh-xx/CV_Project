import math


def adjust_learning_rate(args, optimizer, epoch):
    if args.lr_type == 'step':
        if epoch < 100:
            lr = args.lr
        elif epoch < 140:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    elif args.lr_type == 'cosine':
        warmup_epochs = getattr(args, 'warmup_epochs', 5)
        if epoch < warmup_epochs:
            lr = args.lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        raise KeyError('learning_rate schedule method {} is not achieved')

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

