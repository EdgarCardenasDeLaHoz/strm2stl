import webbrowser
import threading
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import asyncio
import os
import time
import sys

selected_location = {}

# FastAPI initialization (only once)
app = FastAPI()

# Template path using absolute path
templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
print(f"Templates path: {templates_path}")
templates = Jinja2Templates(directory=templates_path)

# Optionally, if you have static files like CSS/JS, uncomment the line below:
# app.mount("/static", StaticFiles(directory="static"), name="static")

class BoundingBox(BaseModel):
    southWestLat: float
    southWestLng: float
    northEastLat: float
    northEastLng: float

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit_bounding_box")
async def submit_bounding_box(bounding_box: BoundingBox):
    print(f"Bounding box received:")
    _ = [print(b) for b in bounding_box]
    # Store the bounding box coordinates in a global variable
    # Process the bounding box coordinates (you can store them, use them for analysis, etc.)
    return {"status": "ok", "bounding_box": bounding_box.dict()}

# Function to run FastAPI server
def run_server():
    uvicorn.run(app, host="127.0.0.1", port=9000)

# Function to detect Jupyter notebook environment
def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPython' in sys.modules:
            return True
        return False
    except ImportError:
        return False

# Function to open browser and select location
def get_location():
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    webbrowser.open("http://127.0.0.1:9000")
    print("Waiting for user to select a bounding box...")

    print(templates_path)

    if not in_notebook():
        # If in Jupyter, use asyncio.sleep for non-blocking operation
        while 'lat' not in selected_location:
            asyncio.run(asyncio.sleep(0.2))
    else:
        # Otherwise, use time.sleep for standard blocking operation
        while 'lat' not in selected_location:
            time.sleep(0.2)

    return selected_location['lat'], selected_location['lng']
