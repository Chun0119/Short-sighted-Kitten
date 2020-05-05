from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

tf.get_logger().setLevel("ERROR")
print("Loading Models...")
IMG_SIZE_FACE = 512
face_detection_model = tf.keras.models.load_model(
    "models/face_detection_model"
).signatures["serving_default"]

IMG_SIZE_LANDMARKS = 224
landmarks_model = tf.keras.models.load_model("models/landmarks_model")
print("Finished Loading ;)")


@app.get("/")
def home():
    return FileResponse("static/index.html")


def detect_face(img):
    img = tf.io.decode_jpeg(img, channels=3)
    shape = tf.shape(img)[:2]  # [height, width]
    img = tf.image.resize_with_pad(img, IMG_SIZE_FACE, IMG_SIZE_FACE)
    img = tf.cast(img, tf.uint8)

    inputs = tf.expand_dims(img, 0)
    outputs = face_detection_model(inputs)
    results = outputs["detections:0"][0]
    bboxes = [bbox[1:5].numpy() for bbox in results]  # x, y, width, height
    bboxes = [
        [bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]]  # x0, y0, x1, y1
        for bbox in bboxes
    ]

    return img, bboxes, shape


def detect_landmarks(img, bbox):
    normalized = tf.constant(bbox) / IMG_SIZE_FACE
    roi = tf.image.crop_and_resize(
        [img],
        [tf.gather(normalized, [1, 0, 3, 2])],  # y0, x0, y1, x1
        [0],
        [IMG_SIZE_LANDMARKS, IMG_SIZE_LANDMARKS],
    )[0]
    roi = (roi / 127.5) - 1  # rescaling to [-1, 1]

    inputs = tf.expand_dims(roi, 0)
    outputs = landmarks_model.predict(inputs)
    landmarks = outputs[0]

    return landmarks


def project_landmarks(shape, bbox, landmarks):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    landmarks = tf.reshape(landmarks, [-1, 2])
    landmarks = landmarks / [IMG_SIZE_LANDMARKS, IMG_SIZE_LANDMARKS] * [width, height]
    landmarks = landmarks + bbox[:2]

    shape = tf.cast(shape, tf.float32)
    base = tf.math.reduce_max(shape)
    edges = (tf.reverse(shape, [-1]) - base) / 2 / base * IMG_SIZE_FACE
    landmarks = landmarks + edges

    landmarks = tf.reshape(landmarks, [-1])
    return landmarks


@app.post("/detect")
def detect(file: UploadFile = File(...)):
    img, bboxes, shape = detect_face(file.file.read())
    landmarks = [detect_landmarks(img, bbox) for bbox in bboxes]
    landmarks = [
        project_landmarks(shape, bbox, landmark)
        for landmark, bbox in zip(landmarks, bboxes)
    ]
    landmarks = [landmark.numpy().tolist() for landmark in landmarks]
    return {"success": True, "landmarks": landmarks}
