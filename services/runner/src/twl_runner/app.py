from fastapi import FastAPI

import twl_engine

app = FastAPI(title="Twinlight runner")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine_version": twl_engine.__version__}
