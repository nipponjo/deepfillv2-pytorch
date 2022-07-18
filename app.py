import uvicorn
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from utils.app_utils import Inpainter

inpainter = Inpainter()
inpainter.load_models('app/models.yaml')

app = FastAPI()


app.mount('/files', StaticFiles(directory='app/files'), 'files')
app.mount('/static', StaticFiles(directory='app/frontend/build/static'), 'static')


@app.get('/')
async def get_page():
    return FileResponse('app/frontend/build/index.html')


@app.get('/{filename}')
async def get_page(filename: str):
    return FileResponse(f'app/frontend/build/{filename}')


@app.get('/api/models')
async def get_model_info():
    model_data = inpainter.get_model_info()
    return {'data': model_data}


@app.post('/api/inpaint')
async def inpaint(image: UploadFile,
                  mask: UploadFile,
                  models: str = Form(),
                  size: int = Form()):
    response_data = inpainter.inpaint(image, mask, models, size)
    return {'data': response_data}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
