# Short-Sighted Kitten

Short-Sighted Kitten is an exploration of facial landmarks detection model on cats. It currently contains pre-trained models for cat face ROI detection and facial landmarks detection, as well as a web application serving as a demo, which adds eyeglasses and speech bubbles to the detected cat eyes and mouth.

![Screenshot](https://user-images.githubusercontent.com/944420/81476487-c5ab7100-9244-11ea-85c3-29ddf11ee72f.jpg)

## Google Colab

The recommended method to try the web application for demo is Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chun0119/Short-sighted-Kitten/blob/master/notebooks/server.ipynb)

## Local Setup

You can also set up the repo in local environment. A GPU instance is recommended as the model loading and inference time may be slow on CPU. Below are the procedures:

1. Install Python 3.6+, preferably via [Anaconda](https://www.anaconda.com/):

```shell
$ python --version
Python 3.7.6
```

2. Install dependencies:

```shell
$ pip install -r requirements.txt
```

3. Download and extracts [pre-trained models](https://drive.google.com/file/d/1ncrxIyUBps_5_iCnYRmRlutCAXrGm4lk/view?usp=sharing) in the project folder:

```shell
$ tree -L 2
...
├── models
│   ├── face_detection_model
│   └── landmarks_model
...
```

4. Launch web server:

```shell
$ uvicorn main:app
```

## Development

- To develop the API server, launch web server with reload:

```shell
$ uvicorn main:app --reload
```

- To develop the client side web application and skip the model loading procedures, launch mock server with reload:

```shell
$ uvicorn mock:app --reload
```

- Launch Chrome Devtools to disable cache

## Models

The models are trained in Google Colab using GPUs. Relevant notebooks are located in the `notebooks` folder:

- `training.ipynb` walks through the training process of the two models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chun0119/Short-sighted-Kitten/blob/master/notebooks/training.ipynb)
- `inference.ipynb` outlines the inference process of the whole pipeline using the saved models, from image to landmarks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chun0119/Short-sighted-Kitten/blob/master/notebooks/inference.ipynb)
- `server.ipynb` describes the procedures to set up the web application on Google Colab. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chun0119/Short-sighted-Kitten/blob/master/notebooks/server.ipynb)
