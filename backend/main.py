import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.search import search_similar_image


app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 2. НАСТРОЙКА CORS
# Список источников (доменов), которым разрешено отправлять запросы
# Ваш фронтэнд на React, вероятно, работает на http://localhost:5173 (Это если че дефолтный порт Vite приложухи для реакта)
origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование папки 'images' для отдачи статических файлов
# Теперь файлы в папке 'images' будут доступны по URL /images/...
app.mount("/images", StaticFiles(directory="images"), name="images")

# Приём файла + вызов search_similar_image
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # вызываем функцию
    result = search_similar_image(file_path)

    # отдаём JSON
    return JSONResponse({"uploaded_file": file_path, "result": result})
