import os
import re
import cv2
import base64
import io
import json
import joblib
import threading
import numpy as np
import tensorflow as tf
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

feature_extractor = None
scaler = None
svm_model = None
mobilenet_extractor = None
_model_cache = {}
_model_lock = threading.Lock()

CATEGORIES = ["Normal", "Reversal"]  # 0 = Normal, 1 = Reversal
THREE_CLASS_NAMES = ["Corrected", "Normal", "Reversal"]
ALL_ASYMMETRIC = ["b", "d", "p", "q", "E", "3", "S", "R", "N", "Z", "J", "2", "5", "7"]
SYMMETRIC_CHARS = set("AHIMOTUVWXYacemnortuvwxz018il")
STATIC_FILES = {"style.css", "script.js", "debug_input.png"}

DEFAULT_MODEL = "cnn_svm"

MODEL_REGISTRY = {
    "cnn_svm": {
        "label": "CNN + SVM",
        "detail": "Recommended. 32x32 CNN features with a binary RBF-SVM.",
        "accuracy": 94.9,
        "recommended": True,
        "uses_orientation_check": True,
    },
    "cnn_rf": {
        "label": "CNN + Random Forest",
        "detail": "32x32 CNN features with a Random Forest classifier.",
        "accuracy": 90.69,
        "recommended": False,
        "uses_orientation_check": False,
    },
    "mobilenet_svm": {
        "label": "MobileNetV2 + SVM",
        "detail": "ImageNet MobileNetV2 features with an SVM classifier.",
        "accuracy": 74.17,
        "recommended": False,
        "uses_orientation_check": False,
    },
    "mobilenet_rf": {
        "label": "MobileNetV2 + Random Forest",
        "detail": "ImageNet MobileNetV2 features with a Random Forest classifier.",
        "accuracy": 73.83,
        "recommended": False,
        "uses_orientation_check": False,
    },
}

MODEL_FILES = {
    "cnn_rf": "final_clean_cnn_random_forest.pkl",
    "mobilenet_svm": "best_mobilenet_svm.pkl",
    "mobilenet_rf": "MobileNetV2_RF_Original.pkl",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feature_extractor, scaler, svm_model
    feature_extractor = tf.keras.models.load_model(os.path.join(BASE_DIR, "feature_extractor.keras"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    svm_model = joblib.load(os.path.join(BASE_DIR, "svm_binary_v2.pkl"))
    yield
    feature_extractor = None
    scaler = None
    svm_model = None
    _model_cache.clear()


app = FastAPI(title="NeuroScript Dyslexia Screening API", lifespan=lifespan)

cnn_dir = os.path.join(BASE_DIR, "cnn")
if os.path.isdir(cnn_dir):
    app.mount("/cnn", StaticFiles(directory=cnn_dir), name="cnn")


def get_classifier(model_key):
    """Loads alternative classifiers on first use to keep startup fast."""
    if model_key in _model_cache:
        return _model_cache[model_key]
    with _model_lock:
        if model_key not in _model_cache:
            path = os.path.join(MODELS_DIR, MODEL_FILES[model_key])
            try:
                # Memory-map the large numpy arrays inside the pickle so the
                # ~300 MB Random Forest models fit on small-memory hosts.
                _model_cache[model_key] = joblib.load(path, mmap_mode="r")
            except Exception as exc:
                label = MODEL_REGISTRY[model_key]["label"]
                raise ValueError(f"Could not load {label}: {exc}") from exc
        return _model_cache[model_key]


def get_mobilenet_extractor():
    global mobilenet_extractor
    if mobilenet_extractor is None:
        with _model_lock:
            if mobilenet_extractor is None:
                mobilenet_extractor = tf.keras.models.load_model(
                    os.path.join(MODELS_DIR, "mobilenet_feature_extractor.keras"),
                    compile=False,
                    safe_mode=False,
                )
    return mobilenet_extractor


def json_error(message, status_code=400):
    return JSONResponse(status_code=status_code, content={"error": message})


def decode_image_safely(file_bytes):
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4] / 255.0
        bgr = img[:, :, :3]
        white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
        return (bgr * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
    return img


def is_valid_handwriting(img_bgr, is_word=False):
    b, g, r = cv2.split(img_bgr)
    color_diff = np.mean(np.abs(r.astype(int) - g.astype(int))) + np.mean(np.abs(g.astype(int) - b.astype(int)))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    colorful_pixels = np.count_nonzero(hsv[:, :, 1] > 45) / (img_bgr.shape[0] * img_bgr.shape[1])

    if color_diff > 15 or colorful_pixels > 0.03:
        return False, "Screenshot or colorful image detected. Please provide clean handwriting input."

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_256 = cv2.resize(gray, (256, 256))

    if np.mean(gray_256) < 127:
        _, binary_256 = cv2.threshold(gray_256, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary_256 = cv2.threshold(gray_256, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_256)
    total_components = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 6)

    max_allowed = 28 if is_word else 8
    if total_components == 0 or total_components > max_allowed:
        return False, "Invalid input: Blank canvas, excessive text, or background clutter detected."

    gray_64 = cv2.resize(gray, (64, 64))
    if np.mean(gray_64) < 127:
        _, binary_64 = cv2.threshold(gray_64, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary_64 = cv2.threshold(gray_64, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ink_ratio = np.count_nonzero(binary_64) / (64 * 64)
    min_ink = 0.008 if is_word else 0.015
    if ink_ratio < min_ink or ink_ratio > 0.45:
        return False, "Invalid input: Blank canvas or non-character object detected."

    return True, "Valid"


def extract_normalized_ink(gray_img):
    """Crops to tight ink bounding box and resizes to 48x48 binary mask (0 or 1)."""
    if np.mean(gray_img) < 127:
        inv = gray_img.copy()
    else:
        inv = 255 - gray_img
    _, mask = cv2.threshold(inv, 50, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    cropped = mask[y:y+h, x:x+w]
    cropped = cv2.dilate(cropped, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_AREA) / 255.0


def render_reference_mask(char):
    """Renders a clean reference glyph of `char` normalized to the same 48x48 box."""
    canvas_ref = np.zeros((120, 120), dtype=np.uint8)
    cv2.putText(canvas_ref, char, (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 8, cv2.LINE_AA)
    coords = cv2.findNonZero(canvas_ref)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    cropped = canvas_ref[y:y+h, x:x+w]
    cropped = cv2.dilate(cropped, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_AREA) / 255.0


def compare_normal_vs_mirror(img_gray, target_char):
    """Compares drawn character against Normal vs. Horizontally Mirrored reference."""
    if target_char and target_char in SYMMETRIC_CHARS:
        return "Normal", 94.0

    drawn = extract_normalized_ink(img_gray)
    if drawn is None:
        return None, None

    norm_d = np.linalg.norm(drawn) + 1e-6
    candidates = [target_char] if target_char else ALL_ASYMMETRIC

    best_score = -1.0
    best_label = None

    for ch in candidates:
        ref_normal = render_reference_mask(ch)
        if ref_normal is None:
            continue
        ref_mirror = cv2.flip(ref_normal, 1)

        score_normal = float(np.sum(drawn * ref_normal) / (norm_d * (np.linalg.norm(ref_normal) + 1e-6)))
        score_mirror = float(np.sum(drawn * ref_mirror) / (norm_d * (np.linalg.norm(ref_mirror) + 1e-6)))

        if score_normal > best_score:
            best_score = score_normal
            best_label = ("Normal", score_normal, score_mirror)
        if score_mirror > best_score:
            best_score = score_mirror
            best_label = ("Reversal", score_mirror, score_normal)

    if best_label is None:
        return None, None

    label, win_score, lose_score = best_label
    conf = min(96.5, max(82.0, 75.0 + (win_score - lose_score) * 120.0))
    return label, round(conf, 2)


def preprocess_character(gray_img):
    """Matches the training pipeline: Otsu threshold, ink-on-black, tight crop, square pad, 32x32."""
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border_pixels = np.concatenate([
        binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]
    ])
    if np.median(border_pixels) > 127:
        binary = 255 - binary

    points = cv2.findNonZero(binary)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        cropped = binary[y:y+h, x:x+w]
        side = max(w, h)
        square = np.zeros((side, side), dtype=np.uint8)
        x_offset = (side - w) // 2
        y_offset = (side - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    else:
        square = binary

    resized = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)
    resized = resized.astype("float32") / 255.0
    return np.expand_dims(resized, axis=-1)


def preprocess_final_cnn(gray_img):
    """CNN+Random Forest preprocessing: white background, dark handwriting, 32x32."""
    _, image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.median(image) < 127:
        image = 255 - image

    coords = cv2.findNonZero(255 - image)
    if coords is None:
        raise ValueError("No handwriting detected by the selected model.")

    x, y, w, h = cv2.boundingRect(coords)
    crop = image[y:y+h, x:x+w]
    side = max(w, h)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = crop

    resized = cv2.resize(canvas, (32, 32)).astype("float32") / 255.0
    return np.expand_dims(resized, axis=(0, -1))


def preprocess_mobilenet(gray_img):
    """MobileNetV2 models were trained on plain 96x96 grayscale, no thresholding."""
    resized = cv2.resize(gray_img, (96, 96), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype("float32")
    return np.expand_dims(preprocess_input(rgb), axis=0)


def model_class_scores(classifier, features, raw_pred=None):
    """Returns per-class confidence percentages for a 3-class classifier."""
    if hasattr(classifier, "predict_proba"):
        try:
            probs = np.asarray(classifier.predict_proba(features)[0], dtype=float)
            if len(probs) == 3:
                return dict(zip(THREE_CLASS_NAMES, (probs * 100.0).round(2).tolist()))
        except (ValueError, NotImplementedError):
            pass

    if hasattr(classifier, "decision_function"):
        try:
            scores = np.asarray(classifier.decision_function(features), dtype=float)
            flat = scores[0] if scores.ndim == 2 else scores
            if len(flat) == 3:
                z = flat * 2.0
                z -= z.max()
                exp = np.exp(z)
                probs = exp / exp.sum()
                return dict(zip(THREE_CLASS_NAMES, (probs * 100.0).round(2).tolist()))
            probability = 1.0 / (1.0 + np.exp(-float(flat[0])))
            return {
                "Corrected": 0.0,
                "Normal": round((1.0 - probability) * 100.0, 2),
                "Reversal": round(probability * 100.0, 2),
            }
        except (ValueError, NotImplementedError):
            pass

    if raw_pred is not None and 0 <= int(raw_pred) < 3:
        scores = {name: 0.0 for name in THREE_CLASS_NAMES}
        scores[THREE_CLASS_NAMES[int(raw_pred)]] = 100.0
        return scores
    return {name: 33.33 for name in THREE_CLASS_NAMES}


def make_confidence_heatmap(class_scores):
    """Builds a seaborn heatmap PNG data URI from per-class confidence scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = list(class_scores.keys())
    values = np.array([[class_scores[label] for label in labels]], dtype=float)

    fig, ax = plt.subplots(figsize=(max(4.6, len(labels) * 1.7), 1.5), dpi=140)
    sns.heatmap(
        values,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar=False,
        vmin=0,
        vmax=100,
        linewidths=1,
        linecolor="white",
        xticklabels=labels,
        yticklabels=["Confidence (%)"],
        annot_kws={"size": 10, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Prediction confidence by class", fontsize=10, pad=7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def predict_with_model_details(img_gray, model_key=DEFAULT_MODEL, target_char=""):
    """Runs only the selected model; returns label, confidence and class scores."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError("Unknown model selected.")

    if model_key == "cnn_svm":
        patch = preprocess_character(img_gray)
        features = feature_extractor.predict(np.expand_dims(patch, axis=0), verbose=0)
        features = scaler.transform(features)

        probabilities = svm_model.predict_proba(features)[0]
        reversal_prob = float(probabilities[1])
        pred_idx = 1 if reversal_prob >= 0.5 else 0
        max_prob = max(reversal_prob, 1.0 - reversal_prob) * 100.0
        class_scores = {
            "Normal": round((1.0 - reversal_prob) * 100.0, 2),
            "Reversal": round(reversal_prob * 100.0, 2),
        }

        if target_char:
            label, sym_conf = compare_normal_vs_mirror(img_gray, target_char)
            if label is not None:
                pred_idx = 0 if label == "Normal" else 1
                max_prob = sym_conf
                other = "Reversal" if pred_idx == 0 else "Normal"
                class_scores = {label: round(max_prob, 2), other: round(100.0 - max_prob, 2)}

        return CATEGORIES[pred_idx], round(max_prob, 2), class_scores

    if model_key == "cnn_rf":
        features = feature_extractor.predict(preprocess_final_cnn(img_gray), verbose=0)
    else:
        features = get_mobilenet_extractor().predict(preprocess_mobilenet(img_gray), verbose=0)

    classifier = get_classifier(model_key)
    raw_pred = int(classifier.predict(features)[0])
    pred_idx = 1 if raw_pred == 2 else 0
    class_scores = model_class_scores(classifier, features, raw_pred)
    confidence = (class_scores["Corrected"] + class_scores["Normal"]) if pred_idx == 0 else class_scores["Reversal"]
    return CATEGORIES[pred_idx], round(confidence, 2), class_scores


def predict_with_model(img_gray, model_key=DEFAULT_MODEL, target_char=""):
    """Backward-compatible wrapper returning only label and confidence."""
    label, confidence, _ = predict_with_model_details(img_gray, model_key, target_char)
    return label, confidence


def classify_single_gray_char(img_gray, source_mode="draw", target_char="", model_key=DEFAULT_MODEL):
    return predict_with_model(img_gray, model_key, target_char)


def segment_word_characters(img_gray):
    """Slices a word image into left-to-right square character patches using connected components."""
    if np.mean(img_gray) < 127:
        _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        clean_gray = 255 - img_gray
    else:
        _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        clean_gray = img_gray.copy()

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= 20 and h >= 8:
            boxes.append([x, y, x + w, y + h])

    if not boxes:
        return []

    boxes.sort(key=lambda b: b[0])
    merged = [boxes[0]]
    for box in boxes[1:]:
        prev = merged[-1]
        overlap = min(prev[2], box[2]) - max(prev[0], box[0])
        min_w = min(prev[2] - prev[0], box[2] - box[0])
        if overlap > 0.35 * min_w:
            prev[0] = min(prev[0], box[0])
            prev[1] = min(prev[1], box[1])
            prev[2] = max(prev[2], box[2])
            prev[3] = max(prev[3], box[3])
        else:
            merged.append(box)

    patches = []
    for (x1, y1, x2, y2) in merged:
        crop = clean_gray[y1:y2, x1:x2]
        cw, ch = x2 - x1, y2 - y1
        side = max(cw, ch)
        pad_x = (side - cw) // 2 + int(side * 0.25)
        pad_y = (side - ch) // 2 + int(side * 0.25)
        square_patch = cv2.copyMakeBorder(
            crop, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255
        )
        patches.append(square_patch)

    return patches


class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"), media_type="text/html")


@app.get("/models")
def available_models():
    return {
        "default": DEFAULT_MODEL,
        "models": [
            {
                "key": key,
                "label": spec["label"] + (" (Recommended)" if spec["recommended"] else ""),
                "detail": spec["detail"],
                "accuracy": spec["accuracy"],
                "recommended": spec["recommended"],
            }
            for key, spec in MODEL_REGISTRY.items()
        ],
    }


@app.post("/login")
def login(data: LoginRequest):
    username = data.username.strip()
    password = data.password.strip()

    if not username or not password:
        return JSONResponse(status_code=400, content={
            "success": False, "message": "Username and password cannot be empty"
        })

    if not re.match("^[a-zA-Z0-9]+$", username) or not re.match("^[a-zA-Z0-9]+$", password):
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": "Special characters are not allowed! Use only letters and numbers."
        })

    return {"success": True}


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    mode: str = Form("upload"),
    target_char: str = Form(""),
    model: str = Form(DEFAULT_MODEL),
):
    source_mode = mode
    target_char = target_char.strip()
    if model not in MODEL_REGISTRY:
        return json_error("Unknown model selected.")

    file_bytes = np.frombuffer(file.file.read(), np.uint8)
    img_bgr = decode_image_safely(file_bytes)
    if img_bgr is None:
        return json_error("Unsupported or corrupted image file.")

    valid, reason = is_valid_handwriting(img_bgr, is_word=False)
    if not valid:
        return json_error(reason)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        prediction, max_prob, class_scores = predict_with_model_details(img_gray, model, target_char)
    except ValueError as exc:
        return json_error(str(exc))

    if max_prob < 55.0:
        return json_error("Low confidence: Image does not clearly resemble a trained handwriting character.")

    spec = MODEL_REGISTRY[model]
    return {
        "prediction": prediction,
        "confidence": f"{max_prob}%",
        "model": model,
        "model_label": spec["label"],
        "class_scores": class_scores,
        "heatmap": make_confidence_heatmap(class_scores),
    }


@app.post("/predict_word")
def predict_word(
    file: UploadFile = File(...),
    target_word: str = Form(""),
    model: str = Form(DEFAULT_MODEL),
):
    target_word = target_word.strip()
    if model not in MODEL_REGISTRY:
        return json_error("Unknown model selected.")

    file_bytes = np.frombuffer(file.file.read(), np.uint8)
    img_bgr = decode_image_safely(file_bytes)
    if img_bgr is None:
        return json_error("Unsupported or corrupted image file.")

    valid, reason = is_valid_handwriting(img_bgr, is_word=True)
    if not valid:
        return json_error(reason)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    patches = segment_word_characters(img_gray)

    if len(patches) < 2:
        return json_error("Could not segment multiple letters. Write with small spaces between letters (print style).")

    results = []
    reversal_count = 0
    collected_scores = []
    for idx, patch in enumerate(patches):
        expected_ch = target_word[idx] if idx < len(target_word) else ""
        try:
            pred, conf, scores = predict_with_model_details(patch, model, expected_ch)
        except ValueError as exc:
            return json_error(str(exc))
        collected_scores.append(scores)
        if pred == "Reversal":
            reversal_count += 1
        results.append({
            "index": idx + 1,
            "expected": expected_ch if expected_ch else f"Char {idx + 1}",
            "prediction": pred,
            "confidence": f"{conf}%"
        })

    overall = "Reversal" if reversal_count > 0 else "Normal"
    avg_conf = round(sum(float(r["confidence"].replace("%", "")) for r in results) / len(results), 2)

    score_keys = THREE_CLASS_NAMES if any("Corrected" in s for s in collected_scores) else CATEGORIES
    class_scores = {
        key: round(sum(s.get(key, 0.0) for s in collected_scores) / len(collected_scores), 2)
        for key in score_keys
    }

    spec = MODEL_REGISTRY[model]
    return {
        "prediction": overall,
        "confidence": f"{avg_conf}%",
        "segmented_count": len(results),
        "reversal_count": reversal_count,
        "characters": results,
        "model": model,
        "model_label": spec["label"],
        "class_scores": class_scores,
        "heatmap": make_confidence_heatmap(class_scores),
    }


@app.get("/{filename}", include_in_schema=False)
def static_file(filename: str):
    if filename not in STATIC_FILES:
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(os.path.join(BASE_DIR, filename))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
