import json
import os
import pickle
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ------------------------------------------------------------
# Step 1. 기본 경로 설정
# ------------------------------------------------------------
# HTML, CSV, 이미지 폴더가 모두 같은 프로젝트 폴더 안에 있습니다.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "celebrities.csv")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.chdir(BASE_DIR)


# ------------------------------------------------------------
# Step 2. CSV 데이터 불러오기
# ------------------------------------------------------------
# 수업에서 배운 pandas 방식으로 데이터를 불러오고 확인합니다.

df = pd.read_csv(DATA_PATH)

print("===== Step 2. celebrity csv 확인 =====")
print(df.head())
print("shape:", df.shape)
print(df.info())
print(df.describe())


# ------------------------------------------------------------
# Step 3. 분석에 사용할 숫자 컬럼 선택
# ------------------------------------------------------------
# 글자 컬럼(name, image, tags)은 제외하고 숫자 컬럼만 사용합니다.

feature_cols = [
    "brightness",
    "warmth",
    "saturation",
    "contrast",
    "clarity",
    "softness",
    "face_ratio",
    "eye_ratio",
    "nose_ratio",
    "mouth_ratio",
]

X = df[feature_cols]

print()
print("===== Step 3. feature 데이터 확인 =====")
print(X.head())
print("X shape:", X.shape)


# ------------------------------------------------------------
# Step 4. KMeans와 PCA 준비
# ------------------------------------------------------------
# 수업에서 배운 KMeans로 연예인 스타일 그룹을 만들고,
# PCA로 2차원 위치도 계산합니다.

kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

print()
print("===== Step 4. KMeans / PCA 결과 확인 =====")
print(df[["name", "cluster", "pca1", "pca2"]].head())
print(df["cluster"].value_counts())

with open("celebrity_model.pkl", "wb") as f:
    pickle.dump({"kmeans": kmeans, "pca": pca, "feature_cols": feature_cols}, f)

print("celebrity_model.pkl 저장 완료")


# ------------------------------------------------------------
# Step 5. 업로드 사진을 숫자 특징으로 바꾸는 함수
# ------------------------------------------------------------
# Pillow 같은 새 라이브러리 대신 matplotlib의 imread와 numpy 계산만 사용합니다.

def image_to_features(image_path):
    img = plt.imread(image_path)

    if img.max() > 1:
        img = img / 255

    if len(img.shape) == 2:
        r = img
        g = img
        b = img
    else:
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

    brightness = gray.mean()
    warmth = ((r - b + 1) / 2).mean()

    max_color = np.maximum(np.maximum(r, g), b)
    min_color = np.minimum(np.minimum(r, g), b)
    saturation = np.where(max_color == 0, 0, (max_color - min_color) / max_color).mean()

    contrast = np.abs(gray - brightness).mean() * 3
    contrast = min(1, contrast)

    clarity = gray.std()
    clarity = min(1, clarity * 4)

    softness = 1 - contrast * 0.7 + brightness * 0.2
    softness = max(0, min(1, softness))

    height, width = gray.shape
    face_ratio = width / height
    face_ratio = max(0, min(1, face_ratio))

    upper = gray[int(height * 0.25):int(height * 0.45), :]
    middle = gray[int(height * 0.40):int(height * 0.65), :]
    lower = gray[int(height * 0.60):int(height * 0.82), :]

    eye_ratio = min(1, upper.std() * 4)
    nose_ratio = min(1, middle.std() * 4)
    mouth_ratio = min(1, lower.std() * 4)

    features = np.array([
        brightness,
        warmth,
        saturation,
        contrast,
        clarity,
        softness,
        face_ratio,
        eye_ratio,
        nose_ratio,
        mouth_ratio,
    ])

    return features


# ------------------------------------------------------------
# Step 6. 사용자 사진과 연예인 데이터 비교
# ------------------------------------------------------------
# numpy로 거리(distance)를 계산합니다. 거리가 작을수록 비슷합니다.

def label_ratio(value, low, mid, high):
    if value < 0.42:
        return low
    if value > 0.62:
        return high
    return mid


def make_geometry(user_features, celeb_features):
    names = ["얼굴형", "눈 비율", "코 비율", "입 비율"]
    labels = [
        ["갸름한 편", "균형형", "넓은 편"],
        ["작은 편", "보통", "큰 편"],
        ["낮은 편", "보통", "뚜렷한 편"],
        ["작은 편", "보통", "큰 편"],
    ]
    user_values = user_features[6:10]
    celeb_values = celeb_features[6:10]
    result = []

    for i in range(4):
        similarity = round((1 - abs(user_values[i] - celeb_values[i])) * 100)
        similarity = max(0, min(100, similarity))
        result.append({
            "label": names[i],
            "user": label_ratio(user_values[i], labels[i][0], labels[i][1], labels[i][2]),
            "celebrity": label_ratio(celeb_values[i], labels[i][0], labels[i][1], labels[i][2]),
            "similarity": similarity,
        })

    return result


def analyze_user_image(image_path):
    user_features = image_to_features(image_path)
    celebrity_features = X.values

    distances = np.sqrt(((celebrity_features - user_features) ** 2).sum(axis=1))
    result_df = df.copy()
    result_df["distance"] = distances
    result_df["score"] = 100 - (result_df["distance"] / result_df["distance"].max() * 45)
    result_df["score"] = result_df["score"].round(0).astype(int)

    user_cluster = int(kmeans.predict([user_features])[0])
    user_pca = pca.transform([user_features])

    print()
    print("===== Step 6. 업로드 사진 분석 결과 =====")
    print("user_features:")
    print(user_features)
    print("user_cluster:", user_cluster)
    print("user_pca:", user_pca)

    result = result_df.sort_values("distance").head(6)

    print()
    print("===== Step 7. 매칭 결과 확인 =====")
    print(result[["name", "score", "distance", "cluster"]])

    matches = []
    for idx in result.index:
        row = result_df.loc[idx]
        celeb_features = X.loc[idx].values
        image = row["image"]
        image_path = os.path.join(BASE_DIR, image.lstrip("/"))

        if not os.path.exists(image_path):
            image = "/assets/celebrities/placeholder.svg"

        reason = "얼굴형, 눈, 코, 입 비율과 전체 색감 수치를 함께 비교했습니다."

        matches.append({
            "name": row["name"],
            "image": image,
            "score": int(row["score"]),
            "reason": reason,
            "cluster": int(row["cluster"]),
            "geometry": make_geometry(user_features, celeb_features),
        })

    with open("celebrity_result.pkl", "wb") as f:
        pickle.dump(result_df, f)

    print("celebrity_result.pkl 저장 완료")

    return {
        "matches": matches,
        "user_cluster": user_cluster,
        "user_pca": [float(user_pca[0, 0]), float(user_pca[0, 1])],
    }


# ------------------------------------------------------------
# Step 8. HTML 업로드 데이터 읽기
# ------------------------------------------------------------
# Python 3.13에서는 cgi가 없어졌기 때문에, 외부 라이브러리 없이
# multipart/form-data에서 photo 파일만 간단히 꺼냅니다.

def parse_photo_upload(headers, body):
    content_type = headers.get("Content-Type", "")
    if "boundary=" not in content_type:
        return None, None

    boundary = content_type.split("boundary=")[1]
    boundary = boundary.encode("utf-8")
    parts = body.split(b"--" + boundary)

    for part in parts:
        if b'name="photo"' not in part:
            continue

        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            return None, None

        header = part[:header_end].decode("utf-8", errors="ignore")
        file_data = part[header_end + 4:]
        file_data = file_data.strip(b"\r\n")

        filename = "upload.jpg"
        if "filename=" in header:
            filename_text = header.split("filename=")[1].split("\r\n")[0]
            filename = filename_text.strip().strip('"')

        return filename, file_data

    return None, None


# ------------------------------------------------------------
# Step 9. HTML과 Python을 연결하는 기본 웹 서버
# ------------------------------------------------------------
# Flask를 사용하지 않고 Python 기본 http.server를 사용합니다.
# HTML에서 사진을 업로드하면 /api/analyze가 분석 결과를 JSON으로 돌려줍니다.

class IntroAIHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        SimpleHTTPRequestHandler.end_headers(self)

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path != "/api/analyze":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        filename, file_data = parse_photo_upload(self.headers, body)

        if file_data is None:
            self.send_json({"error": "photo 파일이 필요합니다."}, 400)
            return

        filename = os.path.basename(filename)
        if filename == "":
            filename = "upload.jpg"

        save_path = os.path.join(UPLOAD_DIR, filename)

        with open(save_path, "wb") as f:
            f.write(file_data)

        try:
            result = analyze_user_image(save_path)
            self.send_json(result, 200)
        except Exception as error:
            print("분석 오류:", error)
            self.send_json({"error": "이미지를 분석할 수 없습니다."}, 400)

    def send_json(self, data, status_code):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


print()
print("===== Step 9. HTML 서버 실행 =====")
print("브라우저에서 http://127.0.0.1:5000 으로 접속하세요.")

server = ThreadingHTTPServer(("127.0.0.1", 5000), IntroAIHandler)
server.serve_forever()
