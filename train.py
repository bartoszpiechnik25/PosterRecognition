from alive_progress import alive_bar
from utils.data_manipulation import PosterDataset, Transform
import torch, os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from typing import List
from utils.model_wrapper import *

if __name__ == '__main__':
    
    train_transform = Transform(transform=PosterDataset.getTrainTransforms())
    val_transform = Transform()
    path = '/home/barti/PosterRecognition'
    dataset_path = path + '/scraper/data/posters'
    train_dataset = PosterDataset(dataset_path,
                                  transform=train_transform)
    val_dataset = PosterDataset(dataset_path,
                                transform=val_transform,
                                split='val')
    test_dataset = PosterDataset(dataset_path,
                                 transform=val_transform,
                                 split='test')
    model = DinoV2ClassifierMLP(num_classes=train_dataset.getNumClasses)
    # model.load_state_dict(torch.load(path + '/checkpoints/DinoV2Classifier_unfrozen_transformer_state_dict.pth').state_dict())
    # model = from_pretrained(path + '/checkpoints/DinoV2Classifier_state_dict.pth')
    # model = torch.load(path + '/checkpoints/DinoV2Classifier_state_dict.pth')

    EPOCHS = 10
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-4,
                                  weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=1,
    #                                             gamma=0.1)
    model.to('cuda')
    writer = SummaryWriter(path + '/checkpoints/logs')
    train_loader = DataLoader(train_dataset,
                              batch_size=512,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4)
    val_loader = DataLoader(val_dataset,
                            batch_size=512,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=512,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=4)
    
    best_val_loss = float('inf')
    best_weights: dict = None

    print(f"Training {model._get_name()} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")

    with alive_bar(EPOCHS, title='Training', dual_line=True) as bar:
        val_loss = float('inf')
        val_acc = .0

        for epoch in range(EPOCHS):

            train_loss, train_acc = training_loop(model,
                                                train_loader,
                                                optimizer,
                                                (val_loss, val_acc),
                                                bar)
            
            val_loss, val_acc = validation_loop(model, val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.state_dict()
            writer.add_scalar('Train/Loss', train_loss, global_step=epoch)
            writer.add_scalar('Train/Accuracy', train_acc, global_step=epoch)

            # Log validation loss and accuracy to Tensorboard
            writer.add_scalar('Validation/Loss', val_loss, global_step=epoch)
            writer.add_scalar('Validation/Accuracy', val_acc, global_step=epoch)
            # scheduler.step()
            bar()

    model.load_state_dict(best_weights)
    test_loss, test_acc = validation_loop(model, test_loader)
    print(f"Test loss: {test_loss:.4f}\nTest accuracy: {test_acc*100:.2f}%")
    saveWeights(model, path + '/checkpoints/DinoV2Classifier_unfrozen_transformer_state_dict.pth')

    
