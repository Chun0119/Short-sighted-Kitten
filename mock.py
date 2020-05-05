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
    # landmarks = [[115.0, 122.0, 179.0, 121.0, 133.0, 169.0]]
    landmarks = [[263.0, 433.0, 379.0, 286.0, 418.0, 466.0]]
    return {"success": True, "landmarks": landmarks}
