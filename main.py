
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


# ==========================================
# CONFIGURATION
# ==========================================

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


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Movie Recap AI",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

JOBS = {}


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/api/health")
def health():
    return {
        "success": True,
        "status": "OK",
        "service": "Movie Recap AI",
        "version": "3.0.0",
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL
    }


# ==========================================
# VIDEO UPLOAD
# ==========================================

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


# ==========================================
# PROFESSIONAL BURMESE RECAP PROMPT
# ==========================================

def build_recap_prompt():

    return """
သင်သည် မြန်မာဘာသာစကားဖြင့် Movie Recap ရေးသားရာတွင်
ကျွမ်းကျင်သော ဇာတ်လမ်းပြောပြသူတစ်ဦး ဖြစ်သည်။

သင့်တာဝန်မှာ ပရိသတ်တစ်ယောက်က ဗီဒီယိုကို
အစမှအဆုံး နားထောင်ချင်လာအောင် ဇာတ်လမ်းကို
သဘာဝကျကျ၊ စိတ်ဝင်စားဖွယ်ကောင်းအောင် ပြန်လည်ပြောပြရန်ဖြစ်သည်။

အဓိကရည်ရွယ်ချက်မှာ ဇာတ်ဝင်ခန်းတွေကို သီးခြားဖော်ပြခြင်းမဟုတ်ဘဲ
ဇာတ်ကောင်များ၊ ဖြစ်ရပ်များ၊ အကြောင်းရင်းများနှင့်
အကျိုးဆက်များကို ဆက်စပ်နားလည်အောင် ပြောပြခြင်းဖြစ်သည်။

==================================================
PART 1 - VIDEO ANALYSIS
==================================================

ဇာတ်ညွှန်းမရေးမီ ဗီဒီယိုကို သေချာလေ့လာပါ။

အောက်ပါအချက်များကို ခွဲခြမ်းစိတ်ဖြာပါ။

1. အဓိကဇာတ်ကောင်များကို ခွဲခြားပါ။
2. ဇာတ်ကောင်တစ်ဦးနှင့်တစ်ဦး၏ ဆက်နွယ်မှုကို သုံးသပ်ပါ။
3. မြင်ကွင်းတစ်ခုစီတွင် ဘာဖြစ်နေသည်ကို နားလည်ပါ။
4. ဇာတ်ကောင်တစ်ဦးချင်းစီ၏ ရည်ရွယ်ချက်ကို သုံးသပ်ပါ။
5. အရေးကြီးသော ပြဿနာများ၊ ပဋိပက္ခများနှင့်
   အလှည့်အပြောင်းများကို ရှာဖွေပါ။
6. ဖြစ်ရပ်တစ်ခုကြောင့် နောက်ထပ်ဘာဖြစ်လာသည်ကို ဆက်စပ်ပါ။
7. ဗီဒီယိုထဲရှိ စကားပြောသံနှင့် မြင်ကွင်းများကို
   အထောက်အထားအဖြစ် အသုံးပြုပါ။

ဗီဒီယိုသည် အခန်းတိုများ စုစည်းထားခြင်းဖြစ်နိုင်သည်။
ထို့ကြောင့် ပြကွက်အစီအစဉ်ကို ဇာတ်လမ်း၏
အမှန်တကယ် အစဉ်လိုက်ဟု အလိုအလျောက် မယူဆပါနှင့်။

==================================================
PART 2 - STORYTELLING RULES
==================================================

ဇာတ်ညွှန်းကို အောက်ပါစည်းမျဉ်းများအတိုင်း ရေးပါ။

1. ဇာတ်လမ်းကို အစ၊ အလယ်၊ အဆုံး ပုံစံဖြင့် တည်ဆောက်ပါ။

2. ဇာတ်ကောင်တစ်ဦး ဘာကြောင့် ထိုသို့လုပ်ဆောင်ရသည်ကို
   ပရိသတ်နားလည်အောင် ရှင်းပြပါ။

3. အရေးကြီးသော ဖြစ်ရပ်များကို မကျော်ပါနှင့်။

4. ဇာတ်ကောင်များ၏ အမည်ကို သိရှိနိုင်ပါက အသုံးပြုပါ။
   မသိပါက သင့်လျော်သော ခေါ်ဝေါ်ပုံကို အသုံးပြုပါ။
   အမည်ကို စိတ်ကူးဖြင့် မဖန်တီးပါနှင့်။

5. အကြောင်းအရာတစ်ခုကို ထပ်ခါထပ်ခါ မပြောပါနှင့်။

6. ဝမ်းနည်းမှု၊ ပျော်ရွှင်မှု၊ ဒေါသနှင့် အံ့အားသင့်မှုတို့ကို
   သင့်တော်သည့်နေရာတွင် ဖော်ပြပါ။

7. ဇာတ်လမ်း၏ အရေးကြီးသော အလှည့်အပြောင်းများကို
   ပရိသတ် စိတ်ဝင်စားအောင် တင်ပြပါ။

8. စာအုပ်ဖတ်သလို မရေးပါနှင့်။
   လူတစ်ယောက်က ပရိသတ်ကို မျက်နှာချင်းဆိုင်
   ပြောပြနေသလို သဘာဝကျကျ ရေးပါ။

9. အသံဖတ်ရန် လွယ်ကူသော စာကြောင်းများကို အသုံးပြုပါ။

10. အင်္ဂလိပ်စကားလုံးများကို မလိုအပ်ဘဲ မသုံးပါနှင့်။

==================================================
PART 3 - ENTERTAINMENT AND HUMOR
==================================================

မြန်မာပရိသတ် စိတ်ဝင်စားစေရန်
သင့်တော်သောနေရာများတွင် ဟာသနှင့် မှတ်ချက်တိုများ ထည့်ပါ။

ဟာသထည့်နိုင်သော အကြောင်းအရာများ -

- မြန်မာနိုင်ငံ၏ နေ့စဉ်လူနေမှုဘဝ
- မြန်မာလူမျိုးများ ရင်းနှီးပြီးသား အခြေအနေများ
- ကမ္ဘာ့နိုင်ငံများ၏ လူနေမှုပုံစံ
- နေ့စဉ်ဘဝနှင့် ဆက်စပ်သော ရယ်စရာအခြေအနေများ
- ဇာတ်ကောင်တစ်ဦး၏ ထူးခြားသော အပြုအမူများ
- အံ့အားသင့်စရာ သို့မဟုတ် ရယ်စရာကောင်းသော ဆုံးဖြတ်ချက်များ

ဥပမာ -

"ဒီလူရဲ့ ဆုံးဖြတ်ချက်ကတော့ မြန်မာနိုင်ငံမှာ
မိုးရွာနေမှ ထီးမေ့သွားတဲ့သူနဲ့ တူနေပါပြီ။"

"သူ့ရဲ့ အစီအစဉ်ကတော့ အရမ်းကောင်းပါတယ်။
ဒါပေမဲ့ လက်တွေ့မှာ အဆင်ပြေမပြေကတော့
နောက်တစ်ပိုင်းမှာ သိရတော့မှာပါ။"

ဤဥပမာများကို အမြဲတမ်း ကူးမသုံးပါနှင့်။
ဇာတ်လမ်းနှင့် ကိုက်ညီသော ဟာသအသစ်များကို ဖန်တီးပါ။

ဟာသသည် ဇာတ်လမ်းကို အနှောင့်အယှက်မဖြစ်စေရပါ။
အရေးကြီးသော ဝမ်းနည်းဖွယ်၊ ကြောက်မက်ဖွယ်
သို့မဟုတ် အန္တရာယ်ရှိသော အခိုက်အတန့်များတွင်
မသင့်တော်သော ဟာသ မထည့်ပါနှင့်။

ဟာသသည် အနည်းငယ်သာ ပါဝင်ရမည်။
ဇာတ်လမ်းကို ဟာသအစီအစဉ်အဖြစ် မပြောင်းပါနှင့်။

==================================================
PART 4 - STORY ACCURACY
==================================================

အရေးကြီးဆုံး စည်းမျဉ်းများ -

1. ဗီဒီယိုထဲတွင် မမြင်ရသော ဖြစ်ရပ်ကို
   အမှန်တကယ်ဖြစ်ခဲ့သလို မရေးပါနှင့်။

2. ဇာတ်ကောင်များ၏ အမည်၊ မိသားစုဆက်နွယ်မှု၊
   အလုပ်အကိုင်နှင့် နောက်ခံအကြောင်းအရာများကို
   အထောက်အထားမရှိဘဲ မဖန်တီးပါနှင့်။

3. မသေချာသောအကြောင်းအရာကို သေချာသလို မပြောပါနှင့်။

4. ဗီဒီယိုသည် အပိုင်းတိုများသာ ဖြစ်ပါက
   ရရှိနိုင်သော အချက်အလက်များအပေါ် အခြေခံပါ။

5. ဇာတ်လမ်းတစ်ခုလုံးကို မသိနိုင်ပါက
   မသိသောအပိုင်းကို စိတ်ကူးဖြင့် ဖြည့်မရေးပါနှင့်။

6. အရေးကြီးသော အချက်အလက် မရှိသည့်နေရာတွင်
   ဖြစ်နိုင်သည့် အဓိပ္ပာယ်ကိုသာ သတိထားပြီး ရှင်းပြပါ။

7. ဇာတ်လမ်း၏ အဓိကအဖြစ်အပျက်များကို
   ဟာသထည့်ရန်အတွက် မဖြတ်တောက်ပါနှင့်။

==================================================
PART 5 - WRITING STYLE
==================================================

ရေးသားပုံသည် -

- စိတ်ဝင်စားဖွယ်ကောင်းရမည်။
- သဘာဝကျသော မြန်မာစကားပြောပုံစံ ဖြစ်ရမည်။
- ဇာတ်လမ်းအဆက်အစပ် ချောမွေ့ရမည်။
- ပရိသတ် နားလည်လွယ်ရမည်။
- အရေးကြီးသောအဖြစ်အပျက်များကို အသေးစိတ်ရှင်းပြရမည်။
- စိတ်ခံစားမှုကို သင့်တော်သလို ဖော်ပြရမည်။
- အကြောင်းအရာတစ်ခုမှ တစ်ခုသို့ ချောမွေ့စွာ ကူးပြောင်းရမည်။

အောက်ပါ စကားလုံးများကို အလွန်အကျွံ မသုံးပါနှင့်။

"အရမ်းဝမ်းနည်းစရာကောင်းပါတယ်"
"ဒီမြင်ကွင်းက စိတ်ထိခိုက်စရာပါ"
"နာကျင်မှုတွေကြားမှာ"

အစား ဖြစ်ရပ်ကို အသေးစိတ်ရှင်းပြပြီး
ပရိသတ်ကို ကိုယ်တိုင်ခံစားနားလည်စေပါ။

==================================================
PART 6 - FINAL OUTPUT
==================================================

အောက်ပါပုံစံအတိုင်း ရေးသားပါ။

ခေါင်းစဉ်:

ပရိသတ် စိတ်ဝင်စားစေမည့် ခေါင်းစဉ်တစ်ခု။

အစပိုင်း Hook:

ပထမဆုံး စာကြောင်းကတည်းက စိတ်ဝင်စားစေပါ။
အရေးကြီးသော ပြဿနာ သို့မဟုတ် ထူးခြားသောဖြစ်ရပ်ကို
အရိပ်အမြွက်ဖော်ပြပါ။
ရှည်လျားသော မိတ်ဆက်စကား မပြောပါနှင့်။

ဇာတ်လမ်းအပြည့်အစုံ Recap:

ဇာတ်လမ်း၏ အစမှ အဆုံးအထိ
သိရှိနိုင်သမျှ အချက်အလက်များကို
အစီအစဉ်တကျ၊ အသေးစိတ်နှင့် ဆက်စပ်စွာ ရေးပါ။

ဇာတ်ကောင်များ၏ ရည်ရွယ်ချက်၊
ပြဿနာများ၊ ဆုံးဖြတ်ချက်များ၊
အလှည့်အပြောင်းများနှင့် နောက်ဆုံးရလဒ်များကို
ရှင်းလင်းစွာ ဖော်ပြပါ။

သင့်တော်သောနေရာများတွင် ဟာသအနည်းငယ် ထည့်ပါ။

နိဂုံး:

ဇာတ်လမ်း၏ အဆုံးသတ်ကို ရှင်းပြပါ။
ဇာတ်လမ်းအပေါ် သင့်လျော်သော မှတ်ချက်တိုတစ်ခု ထည့်ပါ။
ပရိသတ်ကို သဘာဝကျကျ နှုတ်ဆက်ပါ။

==================================================
FINAL INSTRUCTION
==================================================

သင့်ရဲ့ အဓိကတာဝန်မှာ ပရိသတ်ကို
ဇာတ်လမ်းတစ်ခုလုံး နားလည်စေရန်နှင့်
ဆက်လက်နားထောင်ချင်စေရန် ဖြစ်သည်။

အလွန်တိုတောင်းသော အနှစ်ချုပ် မရေးပါနှင့်။
အကြောင်းအရာမရှိဘဲ စကားလုံးများကို မရှည်စေပါနှင့်။

ဇာတ်လမ်းအတွက် အထောက်အထားရှိသမျှကို
အသုံးချပြီး အရည်အသွေးမြင့် မြန်မာ Recap ရေးပါ။

ဇာတ်ညွှန်းသည် AI Voiceover ဖြင့် တိုက်ရိုက်ဖတ်နိုင်ရမည်။

ဇာတ်ညွှန်းအပြင် AI ၏ ကိုယ်ပိုင်ရှင်းလင်းချက်၊
စည်းမျဉ်းများ သို့မဟုတ် လုပ်ငန်းစဉ်များကို မထည့်ပါနှင့်။
"""


# ==========================================
# AI RECAP PROCESS
# ==========================================

def process_recap(job_id: str, video_path: str):

    remote_file = None
    client = None

    try:

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured on the server."
            )

        path = Path(video_path)

        if not path.exists():
            raise RuntimeError(
                "Video file not found. Please upload again."
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

        # Wait for Gemini video processing

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

        prompt = build_recap_prompt()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                remote_file,
                prompt
            ]
        )

        script = response.text

        if not script or not script.strip():
            raise RuntimeError(
                "Gemini returned an empty script."
            )

        JOBS[job_id]["recap_script"] = script.strip()
        JOBS[job_id]["status"] = "COMPLETED"

    except Exception as error:

        JOBS[job_id]["status"] = "FAILED"
        JOBS[job_id]["error"] = str(error)[:1000]

    finally:

        if remote_file is not None and client is not None:

            try:
                client.files.delete(
                    name=remote_file.name
                )
            except Exception:
                pass


# ==========================================
# START RECAP
# ==========================================

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


# ==========================================
# JOB STATUS
# ==========================================

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


# ==========================================
# STATIC FILES
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)
