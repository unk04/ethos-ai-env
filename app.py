from fastapi import FastAPI
from pydantic import BaseModel
from ethical_env_v2 import EthicalEnvV2

app = FastAPI()
env = EthicalEnvV2()

class Action(BaseModel):
    action: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/reset")
def reset(task: str = "easy"):
    global env
    env = EthicalEnvV2(task)
    result = env.reset()
    return {
        "observation": result.observation.__dict__,
        "done": result.done
    }

@app.post("/step")
def step(action: Action):
    result = env.step(action.action)
    return {
        "observation": result.observation.__dict__,
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }
