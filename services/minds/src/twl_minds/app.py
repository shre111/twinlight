from importlib.metadata import version

from fastapi import FastAPI

app = FastAPI(title="Twinlight minds")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": version("twl-minds")}
