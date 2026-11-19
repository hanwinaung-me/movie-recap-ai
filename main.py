
import os
import uuid
import time
import mimetypes
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from google import genai

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

app = FastAPI(
    title="Movie Recap AI",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# MVP job storage. This is temporary memory storage.
JOBS = {}


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "success": True,
        "status": "OK",
        "service": "Movie Recap AI",
        "version": "2.0.0",
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if not original_name or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    if file.content_type and not (
        file.content_type.startswith("video/")
        or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not recognized as a video."
        )

    job_id = str(uuid.uuid4())
    output_path = UPLOAD_DIR / f"{job_id}{extension}"

    total_size = 0

    try:
        with output_path.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Maximum upload size is 100 MB."
                    )

                output.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty."
            )

        JOBS[job_id] = {
            "job_id": job_id,
            "filename": original_name,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "status": "UPLOADED",
            "recap_script": None,
            "error": None
        }

        return {
            "success": True,
            "message": "Video uploaded successfully.",
            "job_id": job_id,
            "filename": original_name,
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "status": "UPLOADED"
        }

    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()


def process_recap(job_id: str, video_path: str):
    remote_file = None

    try:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured on the server."
            )

        path = Path(video_path)

        if not path.exists():
            raise RuntimeError(
                "Video file not found. Please upload the video again."
            )

        JOBS[job_id]["status"] = "SENDING_TO_AI"

        client = genai.Client(api_key=GEMINI_API_KEY)

        mime_type = (
            mimetypes.guess_type(path.name)[0]
            or "video/mp4"
        )

        remote_file = client.files.upload(
            file=str(path),
            config={"mime_type": mime_type}
        )

        JOBS[job_id]["status"] = "AI_PROCESSING"

        # Wait for Gemini to finish processing the video.
        for attempt in range(180):
            current_file = client.files.get(
                name=remote_file.name
            )

            state = (
                current_file.state.name
                if current_file.state
                else "PROCESSING"
            )

            if state == "ACTIVE":
                remote_file = current_file
                break

            if state == "FAILED":
                raise RuntimeError(
                    "Gemini could not process this video."
                )

            time.sleep(5)

        else:
            raise RuntimeError(
                "Gemini video processing timed out."
            )

        JOBS[job_id]["status"] = "WRITING_SCRIPT"

        prompt = """
You are a professional Burmese-language movie recap writer.

Analyze the entire uploaded movie video carefully.

Create an original Burmese movie recap script.

IMPORTANT REQUIREMENTS:

1. Write naturally in Myanmar Unicode.
2. Explain the story in chronological order.
3. Identify the main characters and their relationships.
4. Explain important events, conflicts, twists and consequences.
5. Start with a powerful short hook that makes viewers curious.
6. Use engaging storytelling and natural transitions.
7. Include original commentary and interpretation.
8. Do not copy dialogue or reproduce long passages from the movie.
9. Do not invent scenes or events that are not supported by the video.
10. Keep the narration suitable for AI Burmese voiceover.
11. Avoid English unless a name or technical term requires it.
12. Do not include production notes or instructions to the editor.

OUTPUT FORMAT:

ခေါင်းစဉ်:
[မြန်မာခေါင်းစဉ်]

အစပိုင်း Hook:
[စိတ်ဝင်စားဖွယ် အဖွင့်]

ဇာတ်လမ်းအကျဉ်း:
[အဓိကဇာတ်ကြောင်း]

မြန်မာ Recap ဇာတ်ညွှန်း:
[အသံသွင်းရန် အသင့်ဖြစ်သော ဇာတ်ညွှန်း]

နိဂုံး:
[ဇာတ်လမ်းအပေါ် ကိုယ်ပိုင်သုံးသပ်ချက်]

Use detailed but concise storytelling.
If the uploaded video is only a short clip or incomplete movie,
clearly state that the analysis covers only the available footage.
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[remote_file, prompt]
        )

        script = response.text

        if not script or not script.strip():
            raise RuntimeError(
                "Gemini returned an empty script."
            )

        JOBS[job_id]["recap_script"] = script
        JOBS[job_id]["status"] = "COMPLETED"

    except Exception as error:
        JOBS[job_id]["status"] = "FAILED"
        JOBS[job_id]["error"] = str(error)[:1000]

    finally:
        if remote_file is not None:
            try:
                client.files.delete(name=remote_file.name)
            except Exception:
                pass


@app.post("/api/jobs/{job_id}/recap")
def start_recap(
    job_id: str,
    background_tasks: BackgroundTasks
):
    job = JOBS.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found. Please upload the video again."
        )

    if job["status"] in {
        "SENDING_TO_AI",
        "AI_PROCESSING",
        "WRITING_SCRIPT"
    }:
        return {
            "success": True,
            "message": "Recap is already processing.",
            "job_id": job_id,
            "status": job["status"]
        }

    if job["status"] == "COMPLETED":
        return {
            "success": True,
            "message": "Recap already completed.",
            "job_id": job_id,
            "status": "COMPLETED"
        }

    video_path = None

    for extension in ALLOWED_EXTENSIONS:
        candidate = UPLOAD_DIR / f"{job_id}{extension}"
        if candidate.exists():
            video_path = candidate
            break

    if not video_path:
        raise HTTPException(
            status_code=404,
            detail="Uploaded video not found. Please upload again."
        )

    job["status"] = "QUEUED"
    job["error"] = None

    background_tasks.add_task(
        process_recap,
        job_id,
        str(video_path)
    )

    return {
        "success": True,
        "message": "AI recap started.",
        "job_id": job_id,
        "status": "QUEUED"
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    return {
        "success": True,
        **job
    }


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)
