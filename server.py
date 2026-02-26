from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import base64
from deepshield_v2 import protect_image

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/protect")
async def protect(file: UploadFile = File(...), strength: str = Form("balanced")):

    input_path = "temp_input.jpg"
    output_path = "temp_output.jpg"

    # Save uploaded image
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run DeepShield engine
    results = protect_image(
        input_path=input_path,
        output_path=output_path,
        strength=strength,
        device="cpu"
    )

    # Convert protected image to base64
    with open(output_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")

    return JSONResponse({
        "identity_shift": results["identity_shift"],
        "psnr": results["psnr"],
        "ssim": results["ssim"],
        "fail_rate": results["deepfake_fail_rate"],
        "epsilon": results["epsilon"],
        "level": results["protection_level"],
        "radar_metrics": [0.8,0.85,0.9,0.88,0.83,0.87],  # placeholder
        "protected_image_base64": encoded_string
    })
