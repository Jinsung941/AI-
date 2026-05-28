import json
import os
import pickle
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer

import cv2
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
MODEL_PATH = os.path.join(BASE_DIR, "celebrity_model.pkl")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.chdir(BASE_DIR)


# ------------------------------------------------------------
# Step 2. CSV 데이터 불러오기
# ------------------------------------------------------------
# 수업에서 배운 pandas 방식으로 데이터를 불러오고 확인합니다.

df = pd.read_csv(DATA_PATH)

if "gender" not in df.columns:
    df["gender"] = "all"

feature_cols = sorted([col for col in df.columns if col.startswith("face_pca_")])

for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

face_pca = model["face_pca"]
face_size = model.get("face_size", 64)

print("===== Step 2. celebrity csv 확인 =====")
print(df.head())
print("shape:", df.shape)
print(df.info())
print(df.describe())


# ------------------------------------------------------------
# Step 3. 분석에 사용할 숫자 컬럼 선택
# ------------------------------------------------------------
# 글자 컬럼(name, image, tags)은 제외하고 숫자 컬럼만 사용합니다.

X = df[feature_cols]
feature_mean = X.mean().values
feature_std = X.std().replace(0, 1).values
feature_weights = np.ones(len(feature_cols))

print()
print("===== Step 3. feature 데이터 확인 =====")
print(X.head())
print("X shape:", X.shape)


# ------------------------------------------------------------
# Step 4. KMeans와 PCA 준비
# ------------------------------------------------------------
# 수업에서 배운 KMeans로 연예인 스타일 그룹을 만들고,
# PCA로 2차원 위치도 계산합니다.

cluster_count = min(5, len(df))
kmeans = KMeans(n_clusters=cluster_count, random_state=0, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

if len(df) >= 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]
else:
    pca = None
    df["pca1"] = 0
    df["pca2"] = 0

print()
print("===== Step 4. KMeans / PCA 결과 확인 =====")
print(df[["name", "cluster", "pca1", "pca2"]].head())
print(df["cluster"].value_counts())

print("celebrity_model.pkl 불러오기 완료")


# ------------------------------------------------------------
# Step 5. 업로드 사진을 얼굴 위치/비율 숫자로 바꾸는 함수
# ------------------------------------------------------------
# OpenCV로 얼굴과 눈 위치를 찾고, 코/입 위치는 얼굴 영역 안에서 추정합니다.

face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)
eye_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
)
smile_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml")
)


def load_image_for_cv2(image_path):
    raw = np.fromfile(image_path, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        img = plt.imread(image_path)
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        if len(img.shape) == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
    return bgr


def largest_box(boxes):
    if len(boxes) == 0:
        return None
    return max(boxes, key=lambda box: box[2] * box[3])


def normalized_face_image(image_path):
    bgr = load_image_for_cv2(image_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(60, 60))
    face = largest_box(faces)

    if face is None:
        x, y, fw, fh = 0, 0, width, height
    else:
        x, y, fw, fh = face

    margin_x = int(fw * 0.12)
    margin_y = int(fh * 0.16)
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(width, x + fw + margin_x)
    bottom = min(height, y + fh + margin_y)

    face_img = gray[top:bottom, left:right]
    face_img = cv2.resize(face_img, (face_size, face_size))
    face_img = cv2.equalizeHist(face_img)
    return face_img.astype(np.float32) / 255


def normalized_face_vector(image_path):
    return normalized_face_image(image_path).reshape(1, -1)


def edge_center_ratio(gray_face, y1, y2, x1=0.15, x2=0.85):
    h, w = gray_face.shape
    top = int(h * y1)
    bottom = max(top + 1, int(h * y2))
    left = int(w * x1)
    right = max(left + 1, int(w * x2))
    roi = gray_face[top:bottom, left:right]
    edges = cv2.Canny(roi, 60, 140)
    points = np.argwhere(edges > 0)
    if len(points) == 0:
        return (y1 + y2) / 2, 0.35
    y_mean = points[:, 0].mean() / max(1, roi.shape[0])
    x_spread = (points[:, 1].max() - points[:, 1].min() + 1) / max(1, roi.shape[1])
    return y1 + (y2 - y1) * y_mean, x_spread


def clamp01(value):
    return max(0, min(1, value))


def region_dark_edge(gray_face, y1, y2, x1, x2):
    h, w = gray_face.shape
    top = int(h * y1)
    bottom = max(top + 1, int(h * y2))
    left = int(w * x1)
    right = max(left + 1, int(w * x2))
    roi = gray_face[top:bottom, left:right]
    equalized = cv2.equalizeHist(roi)
    edges = cv2.Canny(equalized, 60, 150)
    dark_ratio = (roi < 70).mean()
    edge_ratio = (edges > 0).mean()
    return dark_ratio, edge_ratio


def accessory_features(gray_face, mouth_width_ratio):
    eye_dark, eye_edge = region_dark_edge(gray_face, 0.20, 0.47, 0.12, 0.88)
    lower_dark, lower_edge = region_dark_edge(gray_face, 0.58, 0.94, 0.18, 0.82)
    hair_dark, hair_edge = region_dark_edge(gray_face, 0.00, 0.24, 0.06, 0.94)
    glasses_score = clamp01((eye_dark - 0.04) * 5.0 + max(0, eye_edge - 0.16) * 1.5)
    beard_score = clamp01((lower_dark - 0.04) * 7.0 + max(0, lower_edge - 0.20) * 0.8)
    hair_style_score = clamp01((hair_dark - 0.05) * 3.8 + max(0, hair_edge - 0.14) * 1.8)

    lower_face = gray_face[int(gray_face.shape[0] * 0.48):, :]
    smiles = smile_cascade.detectMultiScale(
        lower_face,
        scaleFactor=1.7,
        minNeighbors=18,
        minSize=(25, 12),
    )

    if len(smiles) > 0:
        smile_box = largest_box(smiles)
        smile_score = smile_box[2] / max(1, gray_face.shape[1])
    else:
        smile_score = (mouth_width_ratio - 0.35) / 0.45

    return np.array([
        clamp01(glasses_score),
        clamp01(beard_score),
        clamp01(smile_score),
        clamp01(hair_style_score),
    ])


def face_grid_features(gray_face):
    resized = cv2.resize(gray_face, (64, 64))
    equalized = cv2.equalizeHist(resized)
    normalized = equalized.astype(np.float32) / 255
    values = []

    for y in range(8):
        for x in range(8):
            block = normalized[y * 8:(y + 1) * 8, x * 8:(x + 1) * 8]
            values.append(block.mean())

    return np.array(values)

def image_to_features(image_path):
    face_vector = normalized_face_vector(image_path)
    pca_features = face_pca.transform(face_vector)[0]
    if len(pca_features) < len(feature_cols):
        pca_features = np.pad(pca_features, (0, len(feature_cols) - len(pca_features)))
    return pca_features[:len(feature_cols)]


def most_similar_face_part(user_image_path, celebrity_image_path):
    user_face = normalized_face_image(user_image_path)
    celebrity_face = normalized_face_image(celebrity_image_path)
    regions = [
        ("눈 주변", 0.24, 0.48, 0.16, 0.84),
        ("코 중심", 0.40, 0.66, 0.30, 0.70),
        ("입과 턱 라인", 0.58, 0.90, 0.18, 0.82),
        ("얼굴 윤곽", 0.08, 0.92, 0.08, 0.92),
    ]
    best_part = "전체 인상"
    best_distance = None

    for label, y1, y2, x1, x2 in regions:
        h, w = user_face.shape
        top = int(h * y1)
        bottom = max(top + 1, int(h * y2))
        left = int(w * x1)
        right = max(left + 1, int(w * x2))
        user_roi = user_face[top:bottom, left:right]
        celebrity_roi = celebrity_face[top:bottom, left:right]
        distance = np.mean(np.abs(user_roi - celebrity_roi))

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_part = label

    return best_part


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
    names = [
        "얼굴형",
        "눈 사이 거리",
        "눈 크기",
        "눈 위치",
        "코 위치",
        "입 위치",
        "입 너비",
        "기울기 각도",
        "좌우 각도",
        "안경 특징",
        "수염 특징",
        "웃음 모양",
        "머리스타일",
    ]
    labels = [
        ["갸름한 편", "균형형", "넓은 편"],
        ["가까운 편", "보통", "먼 편"],
        ["작은 편", "보통", "큰 편"],
        ["위쪽", "보통", "아래쪽"],
        ["위쪽", "보통", "아래쪽"],
        ["위쪽", "보통", "아래쪽"],
        ["좁은 편", "보통", "넓은 편"],
        ["왼쪽 기울기", "정면", "오른쪽 기울기"],
        ["왼쪽 방향", "정면", "오른쪽 방향"],
        ["거의 없음", "약간 있음", "뚜렷함"],
        ["거의 없음", "약간 있음", "뚜렷함"],
        ["무표정에 가까움", "살짝 웃음", "활짝 웃음"],
        ["밝거나 단순함", "보통", "어둡거나 선명함"],
    ]
    user_values = user_features
    celeb_values = celeb_features
    result = []

    for i in range(len(names)):
        similarity = round((1 - abs(user_values[i] - celeb_values[i])) * 100)
        similarity = max(0, min(100, similarity))
        result.append({
            "label": names[i],
            "user": label_ratio(user_values[i], labels[i][0], labels[i][1], labels[i][2]),
            "celebrity": label_ratio(celeb_values[i], labels[i][0], labels[i][1], labels[i][2]),
            "similarity": similarity,
        })

    return result


def weighted_distance(user_features):
    celebrity_features = X.values
    celebrity_scaled = (celebrity_features - feature_mean) / feature_std
    user_scaled = (user_features - feature_mean) / feature_std
    diff = (celebrity_scaled - user_scaled) * feature_weights
    distances = np.sqrt((diff ** 2).sum(axis=1))
    return distances


def distance_to_score(distance):
    # PCA 얼굴 거리가 가까우면 높게, 멀면 빠르게 낮아지도록 변환합니다.
    # 이전처럼 기본 점수를 높게 깔지 않아 닮지 않은 사진은 낮게 표시됩니다.
    score = 98 * np.exp(-((distance / 10) ** 2))
    score = max(3, min(98, score))
    return round(score)


def best_match_score(distance):
    score = distance_to_score(distance)

    if distance <= 7:
        score = max(score, 88)
    elif distance <= 9:
        score = max(score, 78)
    elif distance <= 11:
        score = max(score, 65)
    elif distance <= 13:
        score = max(score, 52)

    return min(98, score)


def add_display_scores(result_df):
    result_df = result_df.copy()
    base_scores = [distance_to_score(distance) for distance in result_df["distance"]]

    if len(base_scores) == 0:
        result_df["score"] = []
        return result_df

    top_score = best_match_score(float(result_df["distance"].iloc[0]))
    display_scores = []

    for rank, base_score in enumerate(base_scores):
        if rank == 0:
            display_scores.append(top_score)
        elif rank == 1:
            display_scores.append(max(3, min(base_score, top_score - 14)))
        elif rank == 2:
            display_scores.append(max(3, min(base_score, top_score - 24)))
        elif rank == 3:
            display_scores.append(max(3, min(base_score, top_score - 32)))
        else:
            display_scores.append(base_score)

    result_df["score"] = display_scores
    return result_df


def analyze_user_image(image_path):
    user_features = image_to_features(image_path)
    distances = weighted_distance(user_features)
    result_df = df.copy()
    result_df["distance"] = distances

    user_cluster = int(kmeans.predict([user_features])[0])
    if pca is None:
        user_pca = np.array([[0, 0]])
    else:
        user_pca = pca.transform([user_features])

    print()
    print("===== Step 6. 업로드 사진 분석 결과 =====")
    print("user_features:")
    print(user_features)
    print("user_cluster:", user_cluster)
    print("user_pca:", user_pca)

    result = result_df.sort_values("distance").head(4)
    result = add_display_scores(result)

    print()
    print("===== Step 7. 매칭 결과 확인 =====")
    print(result[["name", "score", "distance", "cluster"]])

    matches = []
    for idx in result.index:
        row = result.loc[idx]
        celeb_features = X.loc[idx].values
        image = row["image"]
        celebrity_image_path = os.path.join(BASE_DIR, image.lstrip("/"))

        if not os.path.exists(celebrity_image_path):
            image = "/assets/celebrities/placeholder.svg"
            best_part = "전체 인상"
        else:
            best_part = most_similar_face_part(image_path, celebrity_image_path)

        reason = "가장 닮은 부분: " + best_part

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


def analyze_user_image_with_filter(image_path, gender_filter):
    result = analyze_user_image(image_path)
    if gender_filter == "all":
        return result

    filtered_matches = []
    for match in result["matches"]:
        celeb_gender = df.loc[df["name"] == match["name"], "gender"]
        if len(celeb_gender) > 0 and celeb_gender.iloc[0] == gender_filter:
            filtered_matches.append(match)

    if len(filtered_matches) >= 4:
        result["matches"] = filtered_matches[:4]
        return result

    user_features = image_to_features(image_path)
    distances = weighted_distance(user_features)
    result_df = df.copy()
    result_df["distance"] = distances
    result_df = result_df[result_df["gender"] == gender_filter]
    result_df = result_df.sort_values("distance").head(4)
    result_df = add_display_scores(result_df)

    matches = []
    for idx in result_df.index:
        row = result_df.loc[idx]
        celeb_features = X.loc[idx].values
        image = row["image"]
        image_path_check = os.path.join(BASE_DIR, image.lstrip("/"))
        if not os.path.exists(image_path_check):
            image = "/assets/celebrities/placeholder.svg"
            best_part = "전체 인상"
        else:
            best_part = most_similar_face_part(image_path, image_path_check)
        matches.append({
            "name": row["name"],
            "image": image,
            "score": int(row["score"]),
            "reason": "가장 닮은 부분: " + best_part,
            "cluster": int(row["cluster"]),
            "geometry": make_geometry(user_features, celeb_features),
        })

    result["matches"] = matches
    return result


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


def parse_text_field(headers, body, field_name, default_value):
    content_type = headers.get("Content-Type", "")
    if "boundary=" not in content_type:
        return default_value

    boundary = content_type.split("boundary=")[1].encode("utf-8")
    parts = body.split(b"--" + boundary)
    target = f'name="{field_name}"'.encode("utf-8")

    for part in parts:
        if target not in part:
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            return default_value
        value = part[header_end + 4:].strip(b"\r\n")
        return value.decode("utf-8", errors="ignore")

    return default_value


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
        gender_filter = parse_text_field(self.headers, body, "gender", "all")
        if gender_filter not in ["all", "female", "male"]:
            gender_filter = "all"

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
            result = analyze_user_image_with_filter(save_path, gender_filter)
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


if __name__ == "__main__":
    print()
    print("===== Step 9. HTML 서버 실행 =====")
    print("브라우저에서 http://127.0.0.1:5050 으로 접속하세요.")

    server = ThreadingHTTPServer(("127.0.0.1", 5050), IntroAIHandler)
    server.serve_forever()
