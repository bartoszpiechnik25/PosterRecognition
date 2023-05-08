from alive_progress import alive_bar
from vit.data_manipulation import PosterDataset, Transform
from vit.model import ViTPosterClassifier
import torch, os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List

path = '/home/barti/PosterRecognition'

def plot_learning(title: str, x_label: str, y_label: str, epochs: List[int], **kwargs) -> None:
    plt.figure(figsize=(11,7))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(epochs, kwargs['train_loss'], label="Training loss")
    plt.plot(epochs, kwargs['validation_loss'], label="Validation loss")
    plt.legend()
    plt.savefig(path + f"/vit/{title}.png")


if __name__ == '__main__':
    pth_path = path + '/vit/checkpoints/ViTPosterClassifier_state_dict.pth'
    if os.path.isfile(pth_path):
        model = ViTPosterClassifier.from_pretrained(pth_path)
    else:
        model = ViTPosterClassifier()
    
    train_transform = Transform(transform=PosterDataset.getTrainTransforms())
    val_transform = Transform(transform=PosterDataset.getValTransforms())

    dataset_path = path + '/scraper/data/images'
    train_dataset = PosterDataset(dataset_path,
                                  transform=PosterDataset.getTrainTransforms())
    val_dataset = PosterDataset(dataset_path,
                                transform=PosterDataset.getValTransforms(),
                                split='val')
    test_dataset = PosterDataset(dataset_path,
                                 transform=PosterDataset.getValTransforms(),
                                 split='test')
    EPOCHS = 10
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-4,
                                  weight_decay=1e-4)
    model.to('cuda')

    train_loader = DataLoader(train_dataset,
                              batch_size=256,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4)
    val_loader = DataLoader(val_dataset,
                            batch_size=256,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=256,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=4)
    
    best_val_loss = float('inf')
    best_weights: dict = None
    train_loss = []
    val_loss_list = []
    train_acc = []
    val_acc_list = []

    print(f"Training {model._get_name()} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")
    with alive_bar(EPOCHS, title='Training', dual_line=True) as bar:
        val_loss = float('inf')
        val_acc = .0

        for epoch in range(EPOCHS):
            epoch_loss = .0
            epoch_train_accuracy = .0
            epoch_val_accuracy = .0

            for i, (X, Y) in enumerate(train_loader, 1):
                X = X.to('cuda')
                Y = Y.to('cuda')
                logits, loss = model.forward(X, Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                batch_acc = model.calculateAccuracy(logits, Y, True)
                epoch_train_accuracy += batch_acc

                bar.text = f"Epoch {epoch + 1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Train Accuracy: {batch_acc*100:.2f}% | Val Accuracy: {val_acc*100:.2f}%"
                del X, Y
            val_loss, val_acc = ViTPosterClassifier.validation_loop(model, val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.state_dict()

            train_loss.append(epoch_loss / i)
            train_acc.append(epoch_train_accuracy / i)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            bar()

    model.load_state_dict(best_weights)
    test_loss, test_acc = ViTPosterClassifier.validation_loop(model, test_loader)
    print(f"Test loss: {test_loss:.4f}\nTest accuracy: {test_acc*100:.2f}%")
    model.saveWeights()

    plot_learning("Classfier Learning Curve", "Epochs", "Loss", range(len(val_loss_list)), validation_loss=val_loss_list, train_loss=train_loss)
    plot_learning("Accuracy Learning Curve", "Epochs", "Accuracy", range(len(val_acc_list)), validation_loss=val_acc_list, train_loss=train_acc)