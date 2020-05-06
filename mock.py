from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
from asyncio import sleep

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    await sleep(5)  # simulate inference

    if file.filename == "test1.jpg":
        landmarks = [[115.0, 122.0, 179.0, 121.0, 133.0, 169.0]]
    elif file.filename == "test2.jpg":
        landmarks = [[263.0, 433.0, 379.0, 286.0, 418.0, 466.0]]
    elif file.filename == "test3.jpg":
        landmarks = [
            [408, 694, 507, 681, 458, 749],
            [750, 436, 884, 437, 809, 543],
            [1050, 685, 1152, 691, 1084, 757],
            [1359, 595, 1453, 611, 1380, 666],
        ]
    else:
        landmarks = []

    return {"success": True, "landmarks": landmarks}
