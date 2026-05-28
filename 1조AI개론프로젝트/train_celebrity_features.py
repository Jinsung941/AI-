import os
import pickle
import unicodedata
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "assets", "celebrities_raw")
DATA_PATH = os.path.join(BASE_DIR, "data", "celebrities.csv")
LABEL_PATH = os.path.join(BASE_DIR, "data", "celebrity_labels.csv")
MODEL_PATH = os.path.join(BASE_DIR, "celebrity_model.pkl")
PLOT_PATH = os.path.join(BASE_DIR, "training_pca.png")

FACE_SIZE = 64
PCA_COMPONENTS = 32
FEATURE_COLS = [f"face_pca_{i:02d}" for i in range(PCA_COMPONENTS)]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

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


def normalized_face_vector(image_path):
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
    face_img = cv2.resize(face_img, (FACE_SIZE, FACE_SIZE))
    face_img = cv2.equalizeHist(face_img)
    return (face_img.astype(np.float32) / 255).reshape(-1)


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
    bgr = load_image_for_cv2(image_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(60, 60))
    face = largest_box(faces)

    if face is None:
        x, y, fw, fh = 0, 0, width, height
    else:
        x, y, fw, fh = face

    gray_face = gray[y:y + fh, x:x + fw]
    face_ratio = max(0, min(1, fw / max(1, fh)))

    upper_face = gray_face[0:int(fh * 0.55), :]
    eyes = eye_cascade.detectMultiScale(upper_face, scaleFactor=1.08, minNeighbors=5, minSize=(18, 18))
    eyes = sorted(eyes, key=lambda box: box[2] * box[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda box: box[0])

    if len(eyes) >= 2:
        lx, ly, lw, lh = eyes[0]
        rx, ry, rw, rh = eyes[1]
        left_center = np.array([lx + lw / 2, ly + lh / 2])
        right_center = np.array([rx + rw / 2, ry + rh / 2])
        eye_distance_ratio = np.linalg.norm(right_center - left_center) / max(1, fw)
        eye_size_ratio = ((lw + rw) / 2) / max(1, fw)
        eye_y_ratio = ((left_center[1] + right_center[1]) / 2) / max(1, fh)
        roll_degrees = np.degrees(np.arctan2(right_center[1] - left_center[1], right_center[0] - left_center[0]))
        roll_ratio = 0.5 + roll_degrees / 60
        eye_mid_x = ((left_center[0] + right_center[0]) / 2) / max(1, fw)
        eye_size_asymmetry = (rw - lw) / max(1, (rw + lw) / 2)
        yaw_ratio = 0.5 + (eye_mid_x - 0.5) * 0.8 + eye_size_asymmetry * 0.12
    elif len(eyes) == 1:
        ex, ey, ew, eh = eyes[0]
        eye_distance_ratio = 0.42
        eye_size_ratio = ew / max(1, fw)
        eye_y_ratio = (ey + eh / 2) / max(1, fh)
        roll_ratio = 0.5
        yaw_ratio = (ex + ew / 2) / max(1, fw)
    else:
        eye_distance_ratio = 0.42
        eye_size_ratio = 0.18
        eye_y_ratio = 0.35
        roll_ratio = 0.5
        yaw_ratio = 0.5

    nose_y_ratio, _nose_spread = edge_center_ratio(gray_face, 0.38, 0.64, 0.32, 0.68)
    mouth_y_ratio, mouth_width_ratio = edge_center_ratio(gray_face, 0.58, 0.84, 0.22, 0.78)
    accessory_values = accessory_features(gray_face, mouth_width_ratio)
    grid_features = face_grid_features(gray_face)

    geometry_features = np.array([
        face_ratio,
        clamp01(eye_distance_ratio),
        clamp01(eye_size_ratio),
        clamp01(eye_y_ratio),
        clamp01(nose_y_ratio),
        clamp01(mouth_y_ratio),
        clamp01(mouth_width_ratio),
        clamp01(roll_ratio),
        clamp01(yaw_ratio),
    ])

    return np.concatenate([geometry_features, accessory_values, grid_features])


def train():
    print("===== Step 1. 학습용 폴더 확인 =====")
    print("RAW_DIR:", RAW_DIR)
    print("DATA_PATH:", DATA_PATH)

    celebrity_names = sorted([
        name for name in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, name))
    ])

    print()
    print("===== Step 2. 연예인 폴더 목록 =====")
    print(celebrity_names)
    print("연예인 수:", len(celebrity_names))

    if os.path.exists(LABEL_PATH):
        label_df = pd.read_csv(LABEL_PATH)
        gender_map = {
            unicodedata.normalize("NFC", row["name"]): row.get("gender", "all")
            for _, row in label_df.iterrows()
        }
        tag_map = {
            unicodedata.normalize("NFC", row["name"]): row.get("tags", "학습 데이터")
            for _, row in label_df.iterrows()
        }
    else:
        gender_map = {}
        tag_map = {}

    image_rows = []
    for celebrity_name in celebrity_names:
        display_name = unicodedata.normalize("NFC", celebrity_name)
        celebrity_dir = os.path.join(RAW_DIR, celebrity_name)
        image_files = sorted([
            file_name for file_name in os.listdir(celebrity_dir)
            if os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS
        ])

        print()
        print("연예인:", display_name)
        print("사진 수:", len(image_files))

        first_image_path = ""

        for image_file in image_files:
            image_path = os.path.join(celebrity_dir, image_file)
            try:
                face_vector = normalized_face_vector(image_path)
                print(image_file, face_vector[:10])

                if first_image_path == "":
                    first_image_path = "/assets/celebrities_raw/" + celebrity_name + "/" + image_file

                image_rows.append({
                    "name": display_name,
                    "gender": gender_map.get(display_name, "all"),
                    "image": first_image_path,
                    "tags": tag_map.get(display_name, "학습 데이터"),
                    "file": image_file,
                    "face_vector": face_vector,
                })
            except Exception as error:
                print("읽기 실패:", image_file, error)

        if first_image_path == "":
            print("사용 가능한 사진이 없어서 건너뜁니다.")

    if len(image_rows) == 0:
        print()
        print("학습할 사진이 없습니다.")
        print("assets/celebrities_raw/연예인이름/ 폴더 안에 jpg 또는 png 파일을 넣어주세요.")
        return

    pixel_matrix = np.array([row["face_vector"] for row in image_rows])
    n_components = min(PCA_COMPONENTS, pixel_matrix.shape[0], pixel_matrix.shape[1])
    face_pca = PCA(n_components=n_components)
    pca_matrix = face_pca.fit_transform(pixel_matrix)

    for i in range(PCA_COMPONENTS):
        col = FEATURE_COLS[i]
        if i < n_components:
            for row_index, row in enumerate(image_rows):
                row[col] = pca_matrix[row_index, i]
        else:
            for row in image_rows:
                row[col] = 0.0

    image_df = pd.DataFrame(image_rows)
    rows = []
    for celebrity_name, group in image_df.groupby("name"):
        row = {
            "name": celebrity_name,
            "gender": group["gender"].iloc[0],
            "image": group["image"].iloc[0],
            "tags": group["tags"].iloc[0],
        }
        for col in FEATURE_COLS:
            row[col] = group[col].mean()
        rows.append(row)

    trained_df = pd.DataFrame(rows)

    print()
    print("===== Step 3. 학습 결과 DataFrame 확인 =====")
    print(trained_df.head())
    print("shape:", trained_df.shape)
    print(trained_df.info())
    print(trained_df.describe())

    if trained_df.shape[0] == 0:
        print()
        print("학습할 사진이 없습니다.")
        print("assets/celebrities_raw/연예인이름/ 폴더 안에 jpg 또는 png 파일을 넣어주세요.")
        return

    if os.path.exists(DATA_PATH):
        backup_name = "celebrities_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
        backup_path = os.path.join(BASE_DIR, "data", backup_name)
        old_df = pd.read_csv(DATA_PATH)
        old_df.to_csv(backup_path, index=False, encoding="utf-8-sig")
        print()
        print("기존 celebrities.csv 백업 완료:")
        print(backup_path)

    trained_df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

    print()
    print("===== Step 4. celebrities.csv 저장 완료 =====")
    print(DATA_PATH)

    if trained_df.shape[0] >= 2:
        X = trained_df[FEATURE_COLS]
        cluster_count = min(5, trained_df.shape[0])
        kmeans = KMeans(n_clusters=cluster_count, random_state=0, n_init=10)
        trained_df["cluster"] = kmeans.fit_predict(X)

        display_pca = PCA(n_components=2)
        X_pca = display_pca.fit_transform(X)
        trained_df["pca1"] = X_pca[:, 0]
        trained_df["pca2"] = X_pca[:, 1]

        print()
        print("===== Step 5. KMeans / PCA 확인 =====")
        print(trained_df[["name", "cluster", "pca1", "pca2"]].head())
        print(trained_df["cluster"].value_counts())

        plt.figure(figsize=(8, 6))
        plt.scatter(trained_df["pca1"], trained_df["pca2"], c=trained_df["cluster"])
        plt.title("Trained Celebrity Features PCA")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.savefig(PLOT_PATH)
        plt.close()

        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "kmeans": kmeans,
                "display_pca": display_pca,
                "face_pca": face_pca,
                "feature_cols": FEATURE_COLS,
                "face_size": FACE_SIZE,
                "pca_components": PCA_COMPONENTS,
            }, f)

        print()
        print("===== Step 6. pickle / plot 저장 완료 =====")
        print(MODEL_PATH)
        print(PLOT_PATH)


if __name__ == "__main__":
    train()
