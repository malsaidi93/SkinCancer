from imports import *
from transform_list import transform as T

"""
Augmentations With F1: uncomment line 16-27, and comment tranforms in dataset file to only resize and toTensor
All_ augmentation : Put the tranforms in dataset.py file and comment the limitation of only F1 augmentations

"""

augmentations = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.GaussianBlur((0, 3.0)),
        iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255)),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
    ])
transform = transforms.Compose([
                            # transforms.ToTensor(),
                            transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),  # Random horizontal flip
                            transforms.RandomVerticalFlip(),  # Random vertical flip
                            transforms.RandomRotation(200),
                            # transforms.RandomPerspective(),
                            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),  # Random affine transformation
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
                            transforms.RandomGrayscale(p=0.5),  # Randomly convert to grayscale
                            # transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.5], [0.5])
                            ])

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=8, horizontalalignment='right')
    plt.yticks(tick_marks, class_names, fontsize=8)

    # Normalize the confusion matrix.
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def train_epoch(model, device, dataloader, loss_fn, optimizer, augment_phase, classes_to_augment=[]):
    train_loss, train_correct = 0.0, 0
    model.train()

    for images, labels in dataloader:
        # if args.modality == 'augmented':    
        #     if len(classes_to_augment) > 0:
        #         # print(f'Train_epoch(): Classes_to_augment => {classes_to_augment}')
        #         for idx, label in enumerate(labels.tolist()):

        #             if dataset.class_id[label] in classes_to_augment:
        #                 images[idx] = transform(images[idx])
        # elif args.modality == 'original':
        #     images = images

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()
        # print(f'train_epoch(): Batch => {Counter(labels.cpu().numpy().tolist())}')

    return train_loss,train_correct

def valid_epoch(model,device,dataloader,loss_fn, class_names):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    y_true, y_pred = [], []  # Use for Confusion Matrix
    y_t, y_p = [], []  # Use for Metrics (f1, precision, recall)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        val_correct += (predictions == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

    classes_to_augment = []
    classification_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    LOGGER.info(f'='*20)
    for class_id in classification_rep.keys():
        if class_id in  class_names:
            # if float(classification_rep[class_id]['f1-score']) <= args.threshold_aug:
            if float(classification_rep[class_id]['f1-score']) <= random.randint(0,1):
                classes_to_augment.append(class_id)


    return valid_loss, val_correct, classes_to_augment

def batch_distribution(dataloader):
    combined_batches = []
    for _, label in dataloader:
        combined_batches.extend(label.cpu().numpy().tolist())

    return Counter(combined_batches)

def test_inference(model, device, dataloader, loss_fn, class_names):
    test_loss, test_correct = 0.0, 0
    model.eval()
    y_true, y_pred = [], []  # Use for Confusion Matrix
    y_t, y_p = [], []  # Use for Metrics (f1, precision, recall)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        test_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)

        test_correct += (predictions == labels).sum().item()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

        y_t.append(labels.cpu().numpy())
        y_p.append(predictions.cpu().numpy())

    classification_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    LOGGER.info(f'='*20)
    LOGGER.info(f'Test_inferenece(): Classification Report: \n')
    for key, values in classification_rep.items():
        LOGGER.info(f'{key} => {values}')
    LOGGER.info(f'='*20)

    with open(f'../reports/{args.aug_type}.txt', 'w+') as report:
        report.write(str(classification_rep))

    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_figure = plot_confusion_matrix(cf_matrix, class_names)

    #     wandb.log({"Testing-Confusion-Matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds = y_pred, class_names = class_names)})
    #     wandb.log({"Metrics-Table": wandb.Table(columns=['F1_Score','Precision','Recall'], data=[[f1_s, p_s, r_s]])})

    return test_loss, test_correct, cf_figure, cf_matrix


if __name__ == '__main__':

    args = args_parser()

    # Set device parameter
    if args.gpu:
        if os.name == 'posix' and torch.backends.mps.is_available():  # device is mac m1 chip
            device = 'mps'
        elif os.name == 'nt' and torch.cuda.is_available():  # device is windows with cuda
            device = args.device
        else:
            device = 'cpu'

    # Initialize metrics table to log metrics to wandb
    # metrics_table = wandb.Table(columns=['F1_Score','Precision','Recall'])

    # ======================= DATA ======================= #

    data_dir = '../data/Combined_data/'
    # dataset = SkinCancerWithAugmentation(data_dir, '../csv/train.csv', transform=None)
    dataset = SkinCancer(data_dir, '../csv/train.csv', transform=None)

    dataset_size = len(dataset)
    # test_dataset = SkinCancerWithAugmentation(data_dir, '../csv/test.csv', transform=None)
    test_dataset = SkinCancer(data_dir, '../csv/test.csv', transform=None)
    classes = np.unique(dataset.classes)

    # ======================= Model | Loss Function | Optimizer ======================= #

    if args.model == 'efficientnet':
        model = efficientnet()

    elif args.model == 'resnet':
        model = resnet()

    elif args.model == 'vit':
        model = vit()

    elif args.model == 'convnext':
        model = convnext()

    elif args.model == 'alexnet':
        model = alexnet()

    elif args.model == 'cnn':
        model = cnn()

    # copy weights
    MODEL_WEIGHTS = copy.deepcopy(model.state_dict())

    # ======================= Set Optimizer and loss Function ======================= #
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9)
    elif args.optimizer == 'adamx':
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)

    if args.imbalanced:
        # loss function with class weights
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(dataset.classes),
                                                          y=np.array(dataset.classes_all))
        class_weights = torch.FloatTensor(class_weights).cuda()
        # class_weights = class_weight.compute_class_weight('balanced',classes=np.unique(dataset.classes),y=self.df['dx'].to_numpy()),device='cuda')
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')


    else:
        criterion = nn.CrossEntropyLoss()

    batch_size = args.batch
    class_names = dataset.classes

    # ======================= Logger ======================= #      

    if args.logger == 'tb':
        logger = \
            SummaryWriter(log_dir=f'../tb_logs/{str(datetime.datetime.now().date())}-{model._get_name()}/{args.modality}_{args.epochs}Epochs_{args.aug_type}')

    elif args.logger == 'wb':
        wandb.login(key="7a2f300a61c6b3c4852452a09526c40098020be2")
        logger = wandb.init(
            # Set the project where this run will be logged
            project="SkinCancer_Augmented_CV_UpdateWeights", entity="fau-computer-vision",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "architecture": args.model,
                "dataset": "Skin Cancer",
                "epochs": args.epochs
            })

    else:
        logger = None

    # ======================= Start ======================= #
    start_t = time.time()
    best_acc = 0.0
    step = 0
    k = 5
    splits = KFold(n_splits=k, shuffle=True, random_state=42)

    args.finetune = 'finetune' if args.finetune else 'transfer'

    # ======================= Local Logger ======================= #

    import datetime
    exp_dir = f'../tb_logs/logs/{str(datetime.datetime.now().date())}-{model._get_name()}_{args.modality}_{args.epochs}_{args.aug_type}/'
    os.makedirs(exp_dir, exist_ok=True)
    log_file = f"{exp_dir}/log.log"
    LOGGER = logging.getLogger(__name__)
    setup_logging(log_path=log_file, log_level='INFO', logger=LOGGER)

    # ======================= Local Logger ======================= #
    LOGGER.info(f'Device: {device}')
    augment_phase = False
    
    # Total Epochs = fold * epochs (5*10 = 50)
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        LOGGER.info('Fold: {}, Model: {}'.format(fold, model._get_name()))
        # model.load_state_dict(MODEL_WEIGHTS) # uncomment to start fresh for each fold
        model.to(device)
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)        
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)  # validation
        
        # ======================= Train per fold ======================= #
        
        for epoch in range(args.epochs):
            step += 1
            LOGGER.info(f'Epoch: {epoch + 1}/{args.epochs}')
            start_epoch = time.time()
            
            if augment_phase:
                LOGGER.info(f'Augment: {augment_phase} Classes_to_Augment: {classes_to_augment}')
                train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer, augment_phase, classes_to_augment)
            else:
                train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer, augment_phase)
                

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            end_epoch = time.time()

            LOGGER.info(f'Average Training Loss: {train_loss}')
            LOGGER.info(f'Average Training Acc: {train_acc}')
            LOGGER.info(f'Time/Epoch : {(end_epoch - start_epoch)/60} minutes')
            
            logger.add_scalar('Epoch/Accuracy', train_acc, step)
            logger.add_scalar('Epoch/Loss', train_loss, step)

            # ======================= Save model if new high accuracy ======================= #
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),
                f'../models/Training-{model._get_name()}-{args.modality}-{args.aug_type}-{args.epochs}Epochs-{step}.pth')
            
        # ======================= Test Model on HOS ======================= #
        val_loss, val_correct, classes_to_augment = valid_epoch(model, device, val_loader, criterion, dataset.classes)
        augment_phase = True
        LOGGER.info(f'Augment: {augment_phase} Classes_to_Augment: {classes_to_augment}')
        
        # Validation Metrics
        val_loss = val_loss / len(val_loader.sampler)
        val_acc = val_correct / len(val_loader.sampler) * 100
        
        LOGGER.info(f'Average Validation Loss: {val_loss}')
        LOGGER.info(f'Average Validation acc: {val_acc}')
        logger.add_scalar('Fold/Acc', val_acc, fold)
        logger.add_scalar('Fold/Loss', val_loss, fold)
        
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size) 
    # LOGGER.info(f'Batch Distribution Test: {batch_distribution(test_loader)}')   
    test_loss, test_correct, cf_figure_fold, cf_matrix = test_inference(model, device, test_loader, criterion,
                                                                        class_names)

    test_loss = test_loss / len(test_loader.sampler)
    test_acc = test_correct / len(test_loader.sampler) * 100
    cf_path = f'../output_files/cf_matrix/{model._get_name()}_{args.aug_type}_{args.modality}_Test.npy'
    np.save(cf_path, cf_matrix)

    # logger.add_scalar('Test Acc', test_acc, fold)
    # logger.add_scalar('Test Loss', test_loss, fold)
    
    LOGGER.info(f'Test Acc: {test_acc}')
    LOGGER.info(f'Test Loss: {test_loss}')

    # ======================= Save model if new high accuracy ======================= #
    if test_acc > best_acc:
        LOGGER.info(f'New High Acc: <<<<< {test_acc} >>>>>')
        best_acc = test_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(),
                    f'../models/Test-{model._get_name()}_{args.modality}_{args.aug_type}_{args.epochs}Epochs.pth')

        # Save Scripted Model 
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model,
                        f'../models/Test-scripted_{model._get_name()}_{args.aug_type}_{args.modality}_{args.epochs}Epochs.pt')

    end_train = time.time()
    time_elapsed = start_t - end_train

    LOGGER.info(f'{model._get_name()} Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
