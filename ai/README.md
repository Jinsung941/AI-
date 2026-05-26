# 스타 닮은꼴 봇

`intro_ai_style_bot.py`는 Intro to AI 수업에서 자주 쓰는 방식처럼 `pandas`, `numpy`, `matplotlib`, `pickle`, `PCA`, `KMeans`를 사용해 사진 특징값과 `data/celebrities.csv`의 연예인 수치를 비교합니다.

## 실행

```bash
pip install -r requirements.txt
python3 intro_ai_html_server.py
```

브라우저에서 `http://127.0.0.1:5000`으로 접속한 뒤 HTML 화면에서 사진을 업로드합니다.

## 파일 구조

```text
.
├── app.py
├── index.html
├── intro_ai_html_server.py
├── intro_ai_style_bot.py
├── requirements.txt
├── data/
│   └── celebrities.csv
└── assets/
    └── celebrities/
        └── placeholder.svg
```

## 연예인 사진 추가

`data/celebrities.csv`의 `image` 경로와 같은 이름으로 사진을 넣으면 결과 카드에 표시됩니다.

예시:

```text
assets/celebrities/iu.jpg
assets/celebrities/suzy.jpg
assets/celebrities/cha-eunwoo.jpg
```

CSV는 Excel에서 열어 수정할 수 있습니다. 이 프로젝트의 점수는 실제 얼굴인식이 아니라 제출용 데모에 맞춘 시각 스타일/비율 매칭 점수입니다.
