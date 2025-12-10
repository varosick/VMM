import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates


from src.search import search_similar_image


app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# HTML шаблоны
templates = Jinja2Templates(directory="backend/templates")


# Страница с формой загрузки файла
@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


# Приём файла + вызов search_similar_image
@app.post("/upload")
async def upload_file(file):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # вызываем вашу функцию
    result = search_similar_image(file_path)

    # отдаём JSON
    return JSONResponse({"uploaded_file": file_path, "result": result})
