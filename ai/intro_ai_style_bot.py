import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ------------------------------------------------------------
# Step 1. 연예인 수치 데이터 불러오기
# ------------------------------------------------------------
# 수업에서 배운 것처럼 pandas로 csv 파일을 읽습니다.

df = pd.read_csv("data/celebrities.csv")

print("===== Step 1. 데이터 확인 =====")
print(df.head())
print()
print("shape:", df.shape)
print()
print(df.info())
print()
print(df.describe())


# ------------------------------------------------------------
# Step 2. 비교에 사용할 숫자 컬럼 선택하기
# ------------------------------------------------------------
# CSV 안에는 이름, 이미지 경로, 태그 같은 글자 데이터도 있습니다.
# AI 비교에는 숫자 데이터만 사용합니다.

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
print("===== Step 2. 숫자 데이터 확인 =====")
print(X.head())
print("X shape:", X.shape)


# ------------------------------------------------------------
# Step 3. matplotlib으로 업로드 사진 읽고 수치로 바꾸기
# ------------------------------------------------------------
# Flask, Pillow 같은 새로운 라이브러리는 사용하지 않습니다.
# matplotlib.pyplot.imread()로 이미지를 읽고 numpy 배열로 계산합니다.

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


# 여기에 사용자가 업로드한 사진 경로를 넣으면 됩니다.
# 예시: user_image_path = "my_photo.jpg"
user_image_path = "sample_user.jpg"


# ------------------------------------------------------------
# Step 4. 사용자 사진 분석하기
# ------------------------------------------------------------

print()
print("===== Step 4. 사용자 사진 분석 =====")
print("사용할 이미지 파일:", user_image_path)

try:
    user_features = image_to_features(user_image_path)
    print("user_features:")
    print(user_features)
except FileNotFoundError:
    print("사용자 사진 파일이 아직 없습니다.")
    print("프로젝트 폴더에 sample_user.jpg를 넣거나 user_image_path를 바꿔주세요.")
    user_features = None


# ------------------------------------------------------------
# Step 5. 연예인 수치와 사용자 사진 수치 비교하기
# ------------------------------------------------------------
# numpy를 이용해서 거리(distance)를 계산합니다.
# 거리가 작을수록 더 비슷하다고 볼 수 있습니다.

if user_features is not None:
    celebrity_features = X.values
    distances = np.sqrt(((celebrity_features - user_features) ** 2).sum(axis=1))

    df["distance"] = distances
    df["similarity"] = 100 - (df["distance"] / df["distance"].max() * 45)
    df["similarity"] = df["similarity"].round(2)

    result = df.sort_values("distance").head(6)

    print()
    print("===== Step 5. 매칭 결과 =====")
    print(result[["name", "similarity", "distance", "image"]])


# ------------------------------------------------------------
# Step 6. KMeans로 연예인 스타일 그룹 나누기
# ------------------------------------------------------------
# 수업에서 배운 KMeans를 사용해서 연예인 데이터를 그룹으로 나눕니다.

kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

print()
print("===== Step 6. KMeans 결과 =====")
print(df[["name", "cluster"]].head(20))
print()
print(df["cluster"].value_counts())


# ------------------------------------------------------------
# Step 7. PCA로 2차원 시각화하기
# ------------------------------------------------------------
# 숫자 컬럼이 많기 때문에 PCA로 2차원으로 줄여서 시각화합니다.

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

print()
print("===== Step 7. PCA 결과 =====")
print(df[["name", "pca1", "pca2"]].head())

plt.figure(figsize=(8, 6))
plt.scatter(df["pca1"], df["pca2"], c=df["cluster"])
plt.title("Celebrity Data PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

if user_features is not None:
    user_pca = pca.transform([user_features])
    plt.scatter(user_pca[0, 0], user_pca[0, 1], c="red", marker="x", s=120)
    plt.text(user_pca[0, 0], user_pca[0, 1], "user photo")

plt.show()


# ------------------------------------------------------------
# Step 8. pickle로 결과 저장하기
# ------------------------------------------------------------
# 수업에서 배운 pickle 방식으로 분석된 DataFrame을 저장합니다.

with open("celebrity_result.pkl", "wb") as f:
    pickle.dump(df, f)

print()
print("===== Step 8. pickle 저장 완료 =====")
print("celebrity_result.pkl 파일로 저장했습니다.")
