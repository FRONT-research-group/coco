from dotenv import load_dotenv
from fastapi import FastAPI
from coco.app.controllers import data_controller, lotw_controller

load_dotenv()

app = FastAPI(
    title="Text Regression API",
    version="0.1",
    description="""
    This API provides the Cognitive Coordinator of the SAFE-6G project.

    **Trust functions supported**: Privacy, Reliability, Security, Resilience, Safety
    """,
    contact={
        "name": "George Batsis",
        "email": "gbatsis@iit.demokritos.gr",
    }
)

app.include_router(data_controller.router)
app.include_router(lotw_controller.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
