# Poster-classification

## Idea

The idea of this project is to strenghten my ability to fine-tune a pre-trained classifier on a new dataset.
Dataset consists of over 4700 classes so it will be challenging to train a model that will be able to classify posters into correct classes. Next step will be to create a model that will be generating brief description about the movie based on the poster.

## Training

So far I have created a classifier on ViT (vision transformer) but I'm currently struggling with overfitting.
Main problem I think is that I have too small dataset. So model simply memorizes the data instead of learning how to classify it.

### Loss curve

![Loss curve](vit/Classfier_Learning_Curve.png?raw=true)

### Accuracy curve

![Accuracy curve](vit/Accuracy_Learning_Curve.png?raw=true)

## TODO

- [x] create a dataset
- [x] create a model
- [-] use Bing/Google API to download more posters
- [ ] retraint ViT model on new improved dataset
- [?] try different architecture (Resnet152?)
- [x] create and evaluate prototypical netrwork for few-shot learning
- [ ] try training with different loss (focal loss)
- [ ] create a model that will be generating brief description about the movie based on the poster
- [ ] dockerize the project
- [ ] create a web app that will be using both models
- [ ] deploy the app on AWS/Azure
