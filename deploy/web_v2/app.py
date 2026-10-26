"""EC2 Flask 웹 서버 v2: 장비 설정 필터(8종 × 3단계)를 사진에 걸어 보고, 원본과 필터 적용본을 B512 Endpoint로 예측해 비교한다.

구조는 v1과 같다. 이 서버는 모델과 PyTorch를 갖지 않고, 사진을 Lambda에 보내면 Lambda가 SageMaker Endpoint를 호출한다.
필터는 시험(src/equipment_eval_direct.py)과 같은 코드(cxrfilters/)를 쓰고, 입력은 모델이 보는 512×512 회색조다.
원본과 필터본 둘 다 같은 경로(512 회색조 PNG)로 보내서, 확률 차이가 필터 때문에 생긴 것임을 보장한다.
"""
import base64
import concurrent.futures as cf
import hashlib
import io
import json
import os
import time
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from cxrfilters.equipment_direct import EQUIP, apply_equipment
from cxrfilters.shift_filters import SHIFTS, apply_shift, to_gray

HERE = Path(__file__).resolve().parent
SIZE = 512
SEED = 42
MAX_IMAGE_BYTES = 2 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
CLASS_KO = {"NORMAL": "정상", "BACTERIAL_PNEUMONIA": "세균성 폐렴", "VIRAL_PNEUMONIA": "바이러스성 폐렴"}
CLASS_ORDER = ["NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"]

SAMPLES = {
    "normal": {"label": "정상 예시", "file": "normal.jpeg", "truth": "정상"},
    "bacterial": {"label": "세균성 폐렴 예시", "file": "bacterial.jpeg", "truth": "세균성 폐렴"},
    "viral": {"label": "바이러스성 폐렴 예시", "file": "viral.jpeg", "truth": "바이러스성 폐렴"},
}

# 필터 설명. 단계 값과 배율은 strength_fixed_512.md(시험 전에 고정한 강도)와 측정한 숫자만 쓴다.
CATALOG = [
    {"id": "kvp", "group": "device", "trained": True, "title": "관전압 ↑ (kVp)",
     "happens": "관전압을 높이면 X선이 몸을 더 잘 통과해서 조직 사이의 차이가 줄고 사진이 밋밋해집니다. 뼈처럼 밝은 부위도 덜 하얘집니다.",
     "how": "평균 밝기를 기준으로 밝기 차이(대비)를 줄이고, 밝은 쪽의 기울기를 한 번 더 눌렀습니다. 평균 밝기는 유지합니다. 실제 관전압의 에너지별 대비 변화는 만들 수 없어서 이 두 가지로 근사했습니다.",
     "levels": ["대비 ×0.85 · 밝은 쪽 ×0.8", "대비 ×0.70 · 밝은 쪽 ×0.6", "대비 ×0.60 · 밝은 쪽 ×0.5"],
     "caution": "강 단계는 가운데(척추·심장 부위)가 거의 평평해질 만큼 센 설정입니다."},
    {"id": "dose", "group": "device", "trained": True, "title": "관전류 ↓ (선량 감소)",
     "happens": "관전류나 시간을 줄이면 선량이 줄고, 검출기에 닿는 X선 알갱이(광자)가 적어져서 알갱이 같은 노이즈가 늘어납니다. 소아는 선량을 낮춰 찍는 경우가 많습니다.",
     "how": "밝기에 따라 크기가 다른 노이즈를 더했습니다. 선량이 1/4이 되면 노이즈 크기는 2배(선량의 제곱근에 반비례)입니다. 기준 노이즈는 원본 사진의 노이즈 수준에 맞췄습니다.",
     "levels": ["선량 1/4 (노이즈 약 2배)", "선량 1/16 (노이즈 약 4배)", "선량 1/64 (노이즈 약 8배)"],
     "caution": "약 단계는 눈으로 거의 보이지 않습니다. 노이즈 배율은 원본 val 200장에서 잰 값입니다."},
    {"id": "motion", "group": "device", "trained": True, "title": "노출시간 ↑ (움직임 번짐)",
     "happens": "오래 찍으면 아이가 움직여서 사진이 번집니다. 숨 쉬는 움직임이 있는 아래쪽(횡격막 부근)이 더 크게 번집니다.",
     "how": "세로 방향으로만 번지게 했고, 위쪽은 25%, 아래쪽은 100% 세기로 적용했습니다. 길이는 mm 단위입니다(224 사진에서 1픽셀이 약 1mm).",
     "levels": ["번짐 길이 2mm", "번짐 길이 4mm", "번짐 길이 8mm"],
     "caution": "아래쪽일수록 번짐이 큰 것이 정상입니다."},
    {"id": "scatter", "group": "device", "trained": True, "title": "산란 (격자 없음)",
     "happens": "산란선을 막는 격자 없이 찍으면 흩어진 X선이 안개처럼 깔려서 작은 구조의 대비가 줄어듭니다.",
     "how": "폭 약 25mm로 아주 크게 흐린 사진을 원본에 k배 섞었습니다. 평균 밝기는 유지합니다.",
     "levels": ["산란/직접선 비 k = 0.25", "k = 0.5", "k = 1.0"],
     "caution": "관전압처럼 대비가 줄지만, 안개를 더하는 방식이라 구현이 다릅니다."},
    {"id": "sensor", "group": "device", "trained": True, "title": "센서 화소 크기",
     "happens": "검출기의 화소가 커지면 미세한 무늬가 뭉개집니다.",
     "how": "화소 크기만큼 평균을 낸 뒤 다시 키웠습니다. 화소가 사진 1픽셀 이하면 효과가 없습니다.",
     "levels": ["화소 0.5mm (사진에서 약 1.1픽셀)", "화소 1.0mm (약 2.3픽셀)", "화소 2.0mm (약 4.6픽셀)"],
     "caution": "약 단계는 눈으로 거의 보이지 않습니다. 224 크기에서는 이 차이가 가려져서 512로 시험했습니다."},
    {"id": "sharpen", "group": "post", "trained": False, "title": "경계 강조 (제조사 후처리)",
     "happens": "많은 장비가 촬영 후 경계와 미세 무늬를 또렷하게 만드는 후처리를 자동으로 겁니다. 제조사마다 알고리즘이 다르고 대부분 공개되어 있지 않습니다.",
     "how": "가장 단순한 형태인 언샤프 마스킹으로 대신했습니다. 원본에 (원본 − 흐린 원본)의 α배를 더합니다. 실제 제조사 알고리즘의 재현이 아닙니다.",
     "levels": ["α = 0.5", "α = 1.0", "α = 2.0"],
     "caution": "학습에 섞지 않은 유형입니다. 처음 보는 유형에도 강한지 보기 위해 남겨 둔 필터입니다."},
    {"id": "equalize", "group": "post", "trained": False, "title": "대비 보정 (제조사 후처리)",
     "happens": "장비와 병원마다 자동으로 밝기 분포를 다시 맞추는 대비 보정을 겁니다.",
     "how": "구역별 대비 보정(CLAHE, 8×8 구역)과 사진 전체 밝기 분포 평활화(HE)로 대신했습니다. 강 단계가 HE입니다.",
     "levels": ["CLAHE (클립 1.5)", "CLAHE (클립 3.0)", "HE (전체 평활화)"],
     "caution": "학습에 섞지 않은 유형입니다. CLAHE는 노이즈 추정값을 약 1.7~2.5배로 키우는 것으로 측정되었습니다(224 사진 기준)."},
    {"id": "jpeg", "group": "post", "trained": False, "title": "JPEG 압축 (전송·저장)",
     "happens": "사진을 병원 밖으로 내보내거나 메신저로 보내면 다시 압축되어 블록 무늬가 생기고 미세한 무늬가 사라집니다.",
     "how": "JPEG로 저장했다가 다시 열었습니다. 품질이 낮을수록 손실이 큽니다.",
     "levels": ["품질 75", "품질 40", "품질 20"],
     "caution": "학습에 섞지 않은 유형입니다."},
]
CATALOG_BY_ID = {c["id"]: c for c in CATALOG}
EFFECTS = json.loads((HERE / "data/effects.json").read_text(encoding="utf-8"))


def apply_filter(gray, name, level):
    if level == 0:
        return gray.copy()
    if name in EQUIP:
        return apply_equipment(gray, name, level, seed=SEED)
    return apply_shift(gray, name, level, seed=SEED)


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def data_url(raw):
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def read_upload(body):
    """업로드 사진 검사: 크기, 실제 형식(JPEG/PNG), 디코딩 가능 여부."""
    if not body or len(body) > MAX_IMAGE_BYTES:
        raise ValueError("파일은 0보다 크고 2 MiB 이하여야 합니다.")
    if not (body.startswith(b"\xff\xd8\xff") or body.startswith(b"\x89PNG\r\n\x1a\n")):
        raise ValueError("실제 JPEG 또는 PNG 파일을 선택하세요.")
    try:
        with Image.open(io.BytesIO(body)) as im:
            return to_gray(im, SIZE)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
        raise ValueError("영상 파일을 확인할 수 없습니다.") from e


def create_app(lambda_client=None, function_name=None):
    app = Flask(__name__)
    app.json.sort_keys = False  # 예시 사진 순서(정상, 세균성, 바이러스성)를 그대로 보낸다
    app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 64 * 1024
    function = function_name or os.environ["LAMBDA_FUNCTION_NAME"]
    if lambda_client is None:
        lambda_client = boto3.client("lambda", region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"),
                                     config=Config(connect_timeout=5, read_timeout=95, retries={"total_max_attempts": 1}))

    def predict(raw_png):
        payload = {"content_type": "image/png", "image_base64": base64.b64encode(raw_png).decode("ascii")}
        resp = lambda_client.invoke(FunctionName=function, InvocationType="RequestResponse", Payload=json.dumps(payload).encode("utf-8"))
        result = json.loads(resp["Payload"].read())
        if resp.get("FunctionError") or not result.get("ok"):
            raise RuntimeError(result.get("error", "Lambda 실행 오류"))
        probs = [float(result["probabilities"][c]) for c in CLASS_ORDER]
        k = int(np.argmax(probs))
        return {"probs": probs, "pred": CLASS_ORDER[k], "pred_ko": CLASS_KO[CLASS_ORDER[k]], "model_id": result.get("model_id")}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "EC2 Flask v2", "function": function})

    @app.get("/samples/<sid>.jpeg")
    def sample_file(sid):
        if sid not in SAMPLES:
            return jsonify({"ok": False, "error": "없는 예시입니다."}), 404
        return send_from_directory(HERE / "samples", SAMPLES[sid]["file"], mimetype="image/jpeg")

    @app.get("/api/catalog")
    def catalog():
        return jsonify({"filters": CATALOG, "samples": {k: {"label": v["label"], "truth": v["truth"]} for k, v in SAMPLES.items()},
                        "effects": EFFECTS, "classes": [CLASS_KO[c] for c in CLASS_ORDER]})

    @app.post("/api/run")
    def run():
        name = request.form.get("filter", "")
        try:
            level = int(request.form.get("level", "0"))
        except ValueError:
            level = -1
        if name not in CATALOG_BY_ID or level not in (0, 1, 2, 3):
            return jsonify({"ok": False, "error": "필터 또는 강도 값이 올바르지 않습니다."}), 400
        sample = request.form.get("sample", "")
        try:
            if sample:
                if sample not in SAMPLES:
                    return jsonify({"ok": False, "error": "없는 예시입니다."}), 400
                with Image.open(HERE / "samples" / SAMPLES[sample]["file"]) as im:
                    gray = to_gray(im, SIZE)
                source = {"kind": "sample", "label": SAMPLES[sample]["label"], "truth": SAMPLES[sample]["truth"]}
            else:
                upload = request.files.get("image")
                if upload is None:
                    return jsonify({"ok": False, "error": "예시를 고르거나 image 항목으로 파일을 올리세요."}), 400
                gray = read_upload(upload.read(MAX_IMAGE_BYTES + 1))
                source = {"kind": "upload", "label": "업로드한 사진", "truth": None}
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        filtered = apply_filter(gray, name, level)
        raw_o, raw_f = png_bytes(gray), png_bytes(filtered)
        diff = np.abs(gray.astype(np.int16) - filtered.astype(np.int16))
        t0 = time.perf_counter()
        try:
            with cf.ThreadPoolExecutor(max_workers=2) as ex:
                fo, ff = ex.submit(predict, raw_o), ex.submit(predict, raw_f)
                original, after = fo.result(), ff.result()
        except (BotoCoreError, ClientError, RuntimeError, ValueError):
            app.logger.exception("예측 호출 실패")
            return jsonify({"ok": False, "error": "예측 서버(Lambda/Endpoint) 호출에 실패했습니다."}), 502
        return jsonify({
            "ok": True, "source": source, "filter": name, "level": level,
            "original": {"image": data_url(raw_o), **original}, "filtered": {"image": data_url(raw_f), **after},
            "mean_abs_diff": round(float(diff.mean()), 2), "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            "input_sha256": hashlib.sha256(raw_o).hexdigest()[:12]})

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify({"ok": False, "error": "업로드 크기를 줄이세요. 영상은 최대 2 MiB입니다."}), 413

    return app
