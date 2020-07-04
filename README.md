# Color Extractor KMeans API

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

for dev, quick and easy to just do it the FastAPI way:
```
uvicorn fast_api:app --host 0.0.0.0 --port 8081 --reload
```
the `--reload` flag will auto restart the server with code changes

for prod:
```sh
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5000 fast_api:app
```
