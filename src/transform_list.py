import torchvision.transforms as transforms

transform = [transforms.RandomApply([
    transforms.CenterCrop(224),  # Center crop to a specific size
    transforms.RandomResizedCrop(224),  # Random resized crop with a specific size
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomVerticalFlip(),  # Random vertical flip
    transforms.RandomRotation(10),  # Random rotation by degrees
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),  # Random affine transformation
    transforms.RandomPerspective(),  # Random perspective transformation
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),  # Random erasing
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.RandomGrayscale(p=0.2),  # Randomly convert to grayscale
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with mean and std
])]

