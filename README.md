# Color Extractor KMeans API

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Deployment

### locally
for dev, quick and easy to just do it the FastAPI way:
```
uvicorn fast_api:app --host 0.0.0.0 --port 5001 --reload
```
the `--reload` flag will auto restart the server with code changes

for prod:
```
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5000 fast_api:app
```

### Google App Engine
standard environment use:
```
gcloud app deploy app.yaml
```

### Docker
build the image named _color-extractor_ with
```
docker build -t color-extractor .
```
then run the image as a container with
```
docker run --publish 5001:80 --detach -e TIMEOUT="0" -e GRACEFUL_TIMEOUT="0" --rm --name color-extractor-container color-extractor
```
