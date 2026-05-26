from __future__ import annotations

import csv
import math
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageFilter, ImageStat


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "celebrities.csv"
PLACEHOLDER_IMAGE = "/assets/celebrities/placeholder.svg"
FEATURE_KEYS = ["brightness", "warmth", "saturation", "contrast", "clarity", "softness"]
GEOMETRY_KEYS = ["face_ratio", "eye_ratio", "nose_ratio", "mouth_ratio"]

app = Flask(__name__)


@app.after_request
def add_local_file_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def load_celebrities() -> list[dict]:
    celebrities = []
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            features = [float(row[key]) for key in FEATURE_KEYS]
            geometry = [float(row[key]) for key in GEOMETRY_KEYS]
            celebrities.append(
                {
                    "name": row["name"],
                    "image": row.get("image") or PLACEHOLDER_IMAGE,
                    "tags": row.get("tags", ""),
                    "features": features,
                    "geometry": geometry,
                }
            )
    return celebrities


def extract_visual_features(image: Image.Image) -> list[float]:
    """Extract demo visual-style features, not biometric identity features."""
    image = image.copy()
    image.thumbnail((128, 128))

    pixels = list(image.getdata())
    total = len(pixels)
    if total == 0:
        raise ValueError("empty image")

    brightness_values = []
    warmth_values = []
    saturation_values = []

    for r, g, b in pixels:
        rn, gn, bn = r / 255, g / 255, b / 255
        max_c = max(rn, gn, bn)
        min_c = min(rn, gn, bn)
        brightness = 0.2126 * rn + 0.7152 * gn + 0.0722 * bn
        brightness_values.append(brightness)
        warmth_values.append((rn - bn + 1) / 2)
        saturation_values.append(0 if max_c == 0 else (max_c - min_c) / max_c)

    brightness = sum(brightness_values) / total
    warmth = sum(warmth_values) / total
    saturation = sum(saturation_values) / total
    contrast = min(1, sum(abs(value - brightness) for value in brightness_values) / total * 3)

    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    clarity = min(1, (stat.stddev[0] / 64) if stat.stddev else 0)
    softness = max(0, min(1, 1 - contrast * 0.7 + brightness * 0.2))

    return [brightness, warmth, saturation, contrast, clarity, softness]


def extract_geometry_features(image: Image.Image) -> list[float]:
    """Estimate face-shape and feature-ratio signals with lightweight image zones."""
    gray = image.convert("L").resize((160, 200))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_pixels = list(edges.getdata())
    gray_pixels = list(gray.getdata())
    width, height = gray.size

    active = []
    threshold = max(18, ImageStat.Stat(edges).mean[0] * 1.25)
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if edge_pixels[idx] > threshold and 30 < gray_pixels[idx] < 245:
                active.append((x, y))

    if active:
        xs = [point[0] for point in active]
        ys = [point[1] for point in active]
        box_width = max(xs) - min(xs) + 1
        box_height = max(ys) - min(ys) + 1
        face_ratio = box_width / max(box_height, 1)
    else:
        face_ratio = width / height

    def zone_density(y1: float, y2: float, x1: float = 0.22, x2: float = 0.78) -> float:
        left, right = int(width * x1), int(width * x2)
        top, bottom = int(height * y1), int(height * y2)
        values = []
        for y in range(top, bottom):
            start = y * width + left
            values.extend(edge_pixels[start : y * width + right])
        if not values:
            return 0
        return min(1, (sum(values) / len(values)) / 80)

    eye_ratio = zone_density(0.24, 0.43)
    nose_ratio = zone_density(0.40, 0.62, 0.34, 0.66)
    mouth_ratio = zone_density(0.58, 0.78, 0.28, 0.72)

    return [
        max(0, min(1, face_ratio)),
        max(0, min(1, eye_ratio)),
        max(0, min(1, nose_ratio)),
        max(0, min(1, mouth_ratio)),
    ]


def score_match(uploaded: list[float], celebrity: list[float], index: int) -> int:
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(uploaded, celebrity)))
    wobble = (math.sin(sum(uploaded) * 31.7 + index * 2.13) + 1) * 0.015
    score = (1 - min(distance, 1.35) / 1.35 + wobble) * 100
    return max(55, min(98, round(score)))


def combined_score(
    uploaded_features: list[float],
    celeb_features: list[float],
    uploaded_geometry: list[float],
    celeb_geometry: list[float],
    index: int,
) -> int:
    visual_score = score_match(uploaded_features, celeb_features, index)
    geometry_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(uploaded_geometry, celeb_geometry)))
    geometry_score = max(45, min(98, round((1 - min(geometry_distance, 1.2) / 1.2) * 100)))
    return round(visual_score * 0.58 + geometry_score * 0.42)


def build_reason(features: list[float], tags: str) -> str:
    brightness, warmth, saturation, contrast, _clarity, softness = features
    mood = [
        "밝은 톤" if brightness > 0.58 else "깊은 톤",
        "따뜻한 색감" if warmth > 0.52 else "쿨한 색감",
        "또렷한 분위기" if contrast > 0.43 or saturation > 0.48 else "부드러운 분위기",
    ]
    tag = tags.split("|")[0] if tags else "전체적인 무드"
    return f"{' · '.join(mood)}이 {tag}과 가까워요."


def ratio_label(value: float, low: str, mid: str, high: str) -> str:
    if value < 0.42:
        return low
    if value > 0.62:
        return high
    return mid


def build_geometry_breakdown(uploaded: list[float], celebrity: list[float]) -> list[dict]:
    labels = [
        ("얼굴형", "갸름한 편", "균형형", "넓은 편"),
        ("눈 비율", "작은 편", "보통", "큰 편"),
        ("코 비율", "낮은 편", "보통", "뚜렷한 편"),
        ("입 비율", "작은 편", "보통", "큰 편"),
    ]
    details = []
    for index, (label, low, mid, high) in enumerate(labels):
        user_value = uploaded[index]
        celeb_value = celebrity[index]
        similarity = max(0, min(100, round((1 - abs(user_value - celeb_value)) * 100)))
        details.append(
            {
                "label": label,
                "user": ratio_label(user_value, low, mid, high),
                "celebrity": ratio_label(celeb_value, low, mid, high),
                "similarity": similarity,
            }
        )
    return details


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/assets/<path:path>")
def assets(path: str):
    return send_from_directory(BASE_DIR / "assets", path)


@app.post("/api/analyze")
def analyze():
    uploaded = request.files.get("photo")
    if uploaded is None:
        return jsonify({"error": "photo 파일이 필요합니다."}), 400

    try:
        image = Image.open(uploaded.stream).convert("RGB")
        uploaded_features = extract_visual_features(image)
        uploaded_geometry = extract_geometry_features(image)
    except Exception:
        return jsonify({"error": "이미지를 분석할 수 없습니다."}), 400

    celebrities = load_celebrities()
    matches = []
    for index, celeb in enumerate(celebrities):
        image_path = BASE_DIR / celeb["image"].lstrip("/")
        image = celeb["image"] if image_path.exists() else PLACEHOLDER_IMAGE
        matches.append(
            {
                "name": celeb["name"],
                "image": image,
                "score": combined_score(
                    uploaded_features,
                    celeb["features"],
                    uploaded_geometry,
                    celeb["geometry"],
                    index,
                ),
                "reason": build_reason(uploaded_features, celeb["tags"]),
                "geometry": build_geometry_breakdown(uploaded_geometry, celeb["geometry"]),
            }
        )

    matches.sort(key=lambda item: item["score"], reverse=True)
    return jsonify(
        {
            "disclaimer": "실제 얼굴인식이 아닌 데모용 시각 스타일 매칭입니다.",
            "features": dict(zip(FEATURE_KEYS, [round(value, 4) for value in uploaded_features])),
            "geometry": dict(zip(GEOMETRY_KEYS, [round(value, 4) for value in uploaded_geometry])),
            "matches": matches[:6],
        }
    )


@app.route("/api/analyze", methods=["OPTIONS"])
def analyze_options():
    return ("", 204)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
