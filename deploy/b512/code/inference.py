"""SageMaker Endpoint 내부 코드. 학습 때의 전처리와 클래스 순서를 그대로 쓴다."""
import io
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torchvision import models

CLASSES = ["NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"]
INPUT_SIZE = 512  # 강사 예시는 224. 우리 B512는 학습 때 Resize((512, 512))로 눌렀다 (자르기·패딩 없음)
MAX_BYTES = 2 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


def model_fn(model_dir):
    """컨테이너 시작 시 한 번 실행하여 선택한 가중치를 CPU에 올린다."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["class_names"] != CLASSES or metadata["num_classes"] != 3:
        raise ValueError("모델의 클래스 사전이 다릅니다.")
    expected = {"resize": [INPUT_SIZE, INPUT_SIZE], "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]}
    for key, value in expected.items():
        if metadata["preprocess"][key] != value:
            raise ValueError("학습과 서비스 전처리가 다릅니다: " + key)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    state = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(2)
    return {"model": model, "metadata": metadata}


def input_fn(body, content_type):
    """JPEG 또는 PNG 바이트를 (1, 3, INPUT_SIZE, INPUT_SIZE) 입력 텐서로 변환한다."""
    mime = content_type.split(";")[0].lower()
    if mime not in ("image/jpeg", "image/png"):
        raise ValueError("JPEG 또는 PNG 영상만 보낼 수 있습니다.")
    if not body or len(body) > MAX_BYTES:
        raise ValueError("영상 크기는 0보다 크고 2 MiB 이하여야 합니다.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(body)) as image:
                if image.format not in ("JPEG", "PNG"):
                    raise ValueError("실제 파일 형식이 JPEG 또는 PNG가 아닙니다.")
                image = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
                array = np.asarray(image).transpose(2, 0, 1).copy()
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("영상 파일을 확인할 수 없습니다.") from error
    tensor = torch.from_numpy(array).float().unsqueeze(0) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def predict_fn(tensor, bundle):
    """정답을 읽지 않고 3개 로짓을 Softmax 확률로 바꾼다."""
    start = time.perf_counter()
    with torch.inference_mode():
        logits = bundle["model"](tensor)
        probabilities = torch.softmax(logits.float(), dim=1)[0].tolist()
    prediction = int(np.argmax(probabilities))
    metadata = bundle["metadata"]
    return {"model_id": metadata["exp_id"], "selected_epoch": metadata["selected_epoch"],
            "predicted_label": CLASSES[prediction],
            "probabilities": dict(zip(CLASSES, probabilities)),
            "model_elapsed_ms": round((time.perf_counter() - start) * 1000, 2)}


def output_fn(prediction, accept):
    """HTTP 응답 본문은 UTF-8 JSON이다. 확률과 진단 결과를 구분한다."""
    if accept and accept not in ("application/json", "*/*"):
        raise ValueError("응답 형식은 application/json입니다.")
    return json.dumps(prediction, ensure_ascii=False), "application/json"
