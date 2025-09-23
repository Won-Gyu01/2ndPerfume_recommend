#입력을 버튼이 아닌 선택하는 거로 하는 단계, 한페이지, 나머지는 다 잘 돌아감. 얼굴인식이 잘 안됨
#data = pd.read_csv("C:/PerfumeS/spm/preprocessing_perfumes_dataset1.csv", encoding='cp949')
#model_path = "C:/PerfumeS/spm/runs/detect/train6/weights/best.pt"
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import time

# 모델 경로 설정 (YOLOv8)
model_path = "C:/PerfumeS/spm/runs/detect/train6/weights/best.pt"
model = YOLO(model_path)

# 향수 데이터 로드 및 전처리
def load_and_prepare_data():
    data = pd.read_csv("C:/PerfumeS/spm/preprocessing_perfumes_dataset1.csv", encoding='cp949')
    
    le_dict = {}
    for column in ['department', 'scents', 'concentration']:
        le_dict[column] = LabelEncoder()
        data[column] = le_dict[column].fit_transform(data[column])

    data['base_note_split'] = data['base_note'].str.split(', ')
    data_exploded = data.explode('base_note_split').reset_index(drop=True)
    le_dict['base_note_split'] = LabelEncoder()
    data_exploded['base_note_split'] = le_dict['base_note_split'].fit_transform(data_exploded['base_note_split'])

    data_exploded = data_exploded.drop_duplicates(subset=['name', 'department', 'scents', 'base_note_split', 'concentration'])
    
    return data_exploded, le_dict

# 감정 인식 (카메라 사용)
def detect_mood():
    cap = cv2.VideoCapture(0)
    mood_counts = {'neutral': 0, 'happy': 0, 'sad': 0, 'angry': 0, 'nervous': 0}  # 기분별 카운트 초기화
    start_time = time.time()  # 시작 시간 기록

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture image")
            break
        
        results = model(frame)
        if results and results[0].boxes:
            # 첫 번째 감지된 객체의 클래스 이름을 가져옵니다.
            detected_class = results[0].names[int(results[0].boxes[0].cls)]
            mood_counts[detected_class] = mood_counts.get(detected_class, 0) + 1  # 감지된 기분 카운트 증가
        
        cv2.putText(frame, f"Detecting...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Mood Detection', frame)
        
        # ESC 키를 누르거나 10초가 지나면 종료
        if cv2.waitKey(1) & 0xFF == 27 or time.time() - start_time > 10:
            break

    cap.release()
    cv2.destroyAllWindows()

    # 가장 많이 감지된 기분을 선택
    final_mood = max(mood_counts, key=mood_counts.get)
    return final_mood




# 추천 시스템 로직
def recommend_perfume(gender, preferred_scent, mood, situation, data_exploded, le_dict):
    gender_encoded = le_dict['department'].transform([gender])[0]
    scent_encoded = le_dict['scents'].transform([preferred_scent])[0]
    
    mood_scents = {
        'happy': ['Floral', 'Fruity', 'Citrus'],
        'sad': ['Woody', 'Vanilla', 'Musk'],
        'angry': ['Spicy', 'Fresh', 'Citrus'],
        'nervous': ['Lavender', 'Fresh', 'Floral'],
        'neutral': ['Woody', 'Fresh', 'Musk']
    }.get(mood, [])

    mood_encoded = [le_dict['scents'].transform([s])[0] for s in mood_scents if s in le_dict['scents'].classes_]
    mood_avg = np.mean(mood_encoded) if mood_encoded else 0

    situation_concentration_mapping = {
        'everyday': ['EDT', 'EDC'],
        'special occasion': ['EDP', 'Parfum']
    }
    
    situation_encoded = [le_dict['concentration'].transform([c])[0] for c in situation_concentration_mapping[situation]]
    situation_avg = np.mean(situation_encoded) if situation_encoded else 0

    user_vector = np.array([gender_encoded, scent_encoded, mood_avg, situation_avg])

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = data_exploded[['department', 'scents', 'base_note_split', 'concentration']]
    y = data_exploded['item_rating']
    rf_model.fit(X, y)

    similarity_scores = []
    for name, group in data_exploded.groupby('name'):
        perfume_vector = np.array([group['department'].iloc[0], group['scents'].iloc[0], group['base_note_split'].mean(), group['concentration'].iloc[0]])
        similarity = cosine_similarity(user_vector.reshape(1, -1), perfume_vector.reshape(1, -1))[0][0]

        predicted_rating = rf_model.predict(perfume_vector.reshape(1, -1))[0]
        final_score = similarity * 0.7 + predicted_rating * 0.3

        if group['item_rating'].iloc[0] >= 3.5:
            similarity_scores.append((name, final_score, similarity, group.iloc[0]))

    recommended_perfumes = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]

    return recommended_perfumes

# UI를 통한 사용자 입력 처리
class PerfumeRecommendationApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.data_exploded, self.le_dict = load_and_prepare_data()

    def initUI(self):
        self.layout = QtWidgets.QVBoxLayout()

        # 성별 선택
        self.gender_label = QtWidgets.QLabel("Select Gender:")
        self.gender_combobox = QtWidgets.QComboBox()
        self.gender_combobox.addItems(["Men", "Women", "Unisex"])
        self.layout.addWidget(self.gender_label)
        self.layout.addWidget(self.gender_combobox)

        # 선호 향기 선택
        self.scent_label = QtWidgets.QLabel("Preferred Scent:")
        self.scent_combobox = QtWidgets.QComboBox()
        self.scent_combobox.addItems(["Floral", "Woody", "Fresh", "Spicy", "Citrus", "Lavender", "Musk", "Vanilla", "Fruity"])
        self.layout.addWidget(self.scent_label)
        self.layout.addWidget(self.scent_combobox)

        # 기분 감지 버튼
        self.detect_mood_button = QtWidgets.QPushButton("Detect Mood")
        self.detect_mood_button.clicked.connect(self.detect_mood_and_set)
        self.layout.addWidget(self.detect_mood_button)

        # 감지된 기분 표시
        self.mood_label = QtWidgets.QLabel("Detected Mood:")
        self.detected_mood_label = QtWidgets.QLabel("Not Detected")
        self.layout.addWidget(self.mood_label)
        self.layout.addWidget(self.detected_mood_label)

        # 상황 선택
        self.situation_label = QtWidgets.QLabel("Select Situation:")
        self.situation_combobox = QtWidgets.QComboBox()
        self.situation_combobox.addItems(["everyday", "special occasion"])
        self.layout.addWidget(self.situation_label)
        self.layout.addWidget(self.situation_combobox)

        # 추천 버튼
        self.recommend_button = QtWidgets.QPushButton("Recommend Perfume")
        self.recommend_button.clicked.connect(self.on_click_recommend)
        self.layout.addWidget(self.recommend_button)

        # 결과 표시
        self.result_label = QtWidgets.QLabel("Recommended Perfumes:")
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)
        self.setWindowTitle('Perfume Recommendation System')

    def detect_mood_and_set(self):
        detected_mood = detect_mood()
        self.detected_mood_label.setText(f"Detected Mood: {detected_mood}")

    def on_click_recommend(self):
        gender = self.gender_combobox.currentText()
        preferred_scent = self.scent_combobox.currentText()
        mood = self.detected_mood_label.text().replace("Detected Mood: ", "")  # 감지된 기분을 가져옵니다.
        situation = self.situation_combobox.currentText()

        recommendations = recommend_perfume(gender, preferred_scent, mood, situation, self.data_exploded, self.le_dict)

        result_text = "Recommended Perfumes:\n"
        for perfume in recommendations:
            result_text += f"Brand: {perfume[3]['brand']}\n"
            result_text += f"Name: {perfume[3]['name']}\n"
            result_text += f"Scent: {self.le_dict['scents'].inverse_transform([perfume[3]['scents']])[0]}\n"
            result_text += f"Gender: {self.le_dict['department'].inverse_transform([perfume[3]['department']])[0]}\n"
            result_text += f"Base Note: {perfume[3]['base_note']}\n"
            result_text += f"New Price: {perfume[3]['new_price']}\n"
            result_text += f"Calculated Similarity: {perfume[2]:.4f}\n"
            result_text += f"Final Score: {perfume[1]:.4f}\n"
            result_text += "-" * 30 + "\n"

        self.result_label.setText(result_text)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ex = PerfumeRecommendationApp()
    ex.show()
    sys.exit(app.exec_())
