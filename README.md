# Poster-classification

## Idea

The idea of this project is to strenghten my ability to fine-tune a pre-trained classifier on a new dataset.
Dataset consists of over 4700 classes so it will be challenging to train a model that will be able to classify posters into correct classes. Next step will be to create a model that will be generating brief description about the movie based on the poster.
# ViT Results

## Training

So far I have created a classifier on ViT (vision transformer) but I'm currently struggling with overfitting.
Main problem I think is that I have too small dataset. So model simply memorizes the data instead of learning how to classify it.

### Loss curve

![Loss curve](vit/Classfier_Learning_Curve.png?raw=true)

### Accuracy curve

![Accuracy curve](vit/Accuracy_Learning_Curve.png?raw=true)

# CLIP Results

| Model | Top-1 | Top-3 | Top-5 |
| ----- | ----- | ----- | ----- |
| CLIP Zero-Shot   | 66% | 76% | 80% |

## Done

- 📑 create a dataset
- 📎 create a model
- 🧑‍🔬 experiment with different models
- 🖥️ create REST API for inference

## TODO

- 📝 create a model that will generate brief description about the movie based on the poster
- ❓ fine-tune CLIP for better results
- 🐳 dockerize the project
- 🌩️ deploy the project on Google Cloud Platform (kubernetes ❔)
