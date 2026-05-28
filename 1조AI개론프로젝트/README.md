# Celebrity Lookalike AI Bot

사진을 업로드하면 Python 서버가 얼굴을 분석하고, 학습된 연예인 얼굴 데이터와 비교해서 닮은꼴 Top 4를 보여주는 웹 프로젝트입니다.

## 실행 방법

필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

서버를 실행합니다.

```bash
python3 intro_ai_html_server.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://127.0.0.1:5050
```

## 학습 다시 하기

연예인 사진을 추가하거나 바꾼 뒤 아래 명령어를 실행하면 `data/celebrities.csv`와 `celebrity_model.pkl`이 새로 만들어집니다.

```bash
python3 train_celebrity_features.py
```

사진은 아래 구조로 넣습니다.

```text
assets/celebrities_raw/연예인이름/사진파일.jpg
```

## 현재 분석 방식

OpenCV로 얼굴 영역을 찾은 뒤 얼굴 이미지를 `64x64` 크기로 통일합니다. 그 다음 PCA를 사용해 얼굴 전체 픽셀 패턴을 32개의 숫자 특징으로 줄이고, 연예인별 평균 특징값과 비교합니다.

결과 화면에는 가장 닮은 연예인 Top 4, 전체 유사도 퍼센트, 가장 닮은 얼굴 부분이 표시됩니다.

## 주요 파일

```text
index.html                  웹 화면
intro_ai_html_server.py      업로드/분석 서버
train_celebrity_features.py  연예인 사진 학습 코드
celebrity_model.pkl          PCA 모델 파일
data/celebrities.csv         연예인별 학습 특징값
data/celebrity_labels.csv    성별/태그 정보
assets/celebrities_raw/      연예인 원본 사진 폴더
assets/celebrities/          기본 이미지 폴더
```
