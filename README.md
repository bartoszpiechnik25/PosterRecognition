# Poster-classification

## Idea

The idea of this project is to strenghten my ability to fine-tune a pre-trained classifier on a new dataset.
Dataset consists of over 4700 classes so it will be challenging to train a model that will be able to classify posters into correct classes. Next step will be to create a model that will be generating brief description about the movie based on the poster.

## Try it out

To try it out simply use curl with POST request to the following endpoint:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"url": "https://www.themoviedb.org/t/p/original/2mRRFbnMPMSh4ZiRdiAK0q303Nm.jpg"}' http://34.122.59.71:80/predict
```
In this example I provided url to an image of the movie poster. You can also provide base64 encoded image in the request body, it would look like this:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"base64": "base64_string_encoding_image"}' http://34.122.59.71:80/predict
```

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
- 🐳 dockerize the project
- 🌩️ deploy the project on Google Cloud Platform (Google Kubernetes Engine)


## TODO

- 📝 create a model that will generate brief description about the movie based on the poster
- ❓ fine-tune CLIP for better results
