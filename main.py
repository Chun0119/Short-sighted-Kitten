from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

tf.get_logger().setLevel("ERROR")
print("Loading Models...")
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
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    shape = tf.shape(img)[:2]  # [height, width]
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


def detect_landmarks(img, shape, bbox):
    dimensions = tf.reverse(shape, axis=[-1])
    dimensions = tf.cast(dimensions, tf.float32)
    normalized = tf.reshape(bbox, [-1, 2]) / dimensions
    normalized = tf.reshape(normalized, [-1])
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
    # project landmarks from IMG_SIZE_LANDMARKS to bbox
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    landmarks = tf.reshape(landmarks, [-1, 2])
    landmarks = landmarks / [IMG_SIZE_LANDMARKS, IMG_SIZE_LANDMARKS] * [width, height]

    # shift landmarks wrt bbox in original image
    landmarks = landmarks + bbox[:2]

    landmarks = tf.reshape(landmarks, [-1])
    return landmarks

def detect_cat(raw):
    img, bboxes, shape = detect_face(raw)
    landmarks = [detect_landmarks(img, shape, bbox) for bbox in bboxes]
    landmarks = [
        project_landmarks(shape, bbox, landmark)
        for landmark, bbox in zip(landmarks, bboxes)
    ]
    landmarks = [landmark.numpy().tolist() for landmark in landmarks]
    return landmarks

@app.post("/detect")
def detect(file: UploadFile = File(...)):
    raw = file.file.read()
    landmarks = detect_cat(raw)
    return {"success": True, "landmarks": landmarks}
