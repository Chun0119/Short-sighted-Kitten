# Short-Sighted Kitten

Short-Sighted Kitten is an exploration of facial landmarks detection model on cats. It currently contains pretrained models for cat face ROI detection and facial landmarks detection, as well as a web application serving as a demo, which adds eyeglasses and speech bubbles to the detected cat eyes and mouth.

## Setup

1. Install Python 3.6+, preferably via [Anaconda](https://www.anaconda.com/):

```shell
$ python --version
Python 3.7.6
```

2. Install dependencies:

```shell
$ pip install -r requirements.txt
```

3. Download and extracts [pretrained models](https://drive.google.com/file/d/1ncrxIyUBps_5_iCnYRmRlutCAXrGm4lk/view?usp=sharing) in the project folder:

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

## Model Training

The models are trained in Google Colab using GPUs. Details can be found in the `notebooks` folder.
