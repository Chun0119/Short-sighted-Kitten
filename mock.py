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
    # landmarks = []
    # landmarks = [[115.0, 122.0, 179.0, 121.0, 133.0, 169.0]]
    # landmarks = [[263.0, 433.0, 379.0, 286.0, 418.0, 466.0]]
    landmarks = [
        [408, 694, 507, 681, 458, 749],
        [750, 436, 884, 437, 809, 543],
        [1050, 685, 1152, 691, 1084, 757],
        [1359, 595, 1453, 611, 1380, 666],
    ]
    return {"success": True, "landmarks": landmarks}
