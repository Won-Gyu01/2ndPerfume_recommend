#얼굴 인식 없이 pyqt를 활용하여 나타냄 잘 작동함
import sys
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot  # 여기에 추가
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


class PerfumeRecommendationApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_data()
        self.prepare_model()

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

        # 기분 선택
        self.mood_label = QtWidgets.QLabel("Select Mood:")
        self.mood_combobox = QtWidgets.QComboBox()
        self.mood_combobox.addItems(["happy", "sad", "angry", "nervous", "neutral"])
        self.layout.addWidget(self.mood_label)
        self.layout.addWidget(self.mood_combobox)

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

    def load_data(self):
        # 데이터 로드 및 전처리
        self.data = pd.read_csv("C:/PerfumeS/spm/preprocessing_perfumes_dataset1.csv",encoding='cp949')

        # 범주형 변수 인코딩
        self.le_dict = {}
        for column in ['department', 'scents', 'concentration']:
            self.le_dict[column] = LabelEncoder()
            self.data[column] = self.encode_label(self.data[column], self.le_dict[column], is_fit=True)

        # base_note 인코딩
        self.data['base_note_split'] = self.data['base_note'].str.split(', ')
        self.data_exploded = self.data.explode('base_note_split').reset_index(drop=True)
        self.le_dict['base_note_split'] = LabelEncoder()
        self.data_exploded['base_note_split'] = self.encode_label(self.data_exploded['base_note_split'], self.le_dict['base_note_split'], is_fit=True)

        # 중복 제거 및 데이터 정리
        self.data_exploded = self.data_exploded.drop_duplicates(subset=['name', 'department', 'scents', 'base_note_split', 'concentration'])

    def encode_label(self, label, encoder, is_fit=False):
        if is_fit:
            return encoder.fit_transform(label)
        else:
            try:
                return encoder.transform([label])[0]
            except ValueError:
                return -1  # 알 수 없는 레이블은 -1 반환

    def prepare_model(self):
        # 기분에 따른 향 매핑
        self.mood_scent_mapping = {
            'happy': ['Floral', 'Fruity', 'Citrus'],
            'sad': ['Woody', 'Vanilla', 'Musk'],
            'angry': ['Spicy', 'Fresh', 'Citrus'],
            'nervous': ['Lavender', 'Fresh', 'Floral'],
            'neutral': ['Woody', 'Fresh', 'Musk']
        }

        # 상황에 따른 향 농도 매핑
        self.situation_concentration_mapping = {
            'everyday': ['EDT', 'EDC'],
            'special occasion': ['EDP', 'Parfum']
        }

        # 랜덤 포레스트 모델 학습
        X = self.data_exploded[['department', 'scents', 'base_note_split', 'concentration']]
        y = self.data_exploded['item_rating']
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)

    def calculate_similarity(self, user_vector, perfume_vector):
        return cosine_similarity(user_vector.reshape(1, -1), perfume_vector.reshape(1, -1))[0][0]

    def recommend_perfume(self, gender, preferred_scent, mood, situation):
        # 사용자 입력 인코딩
        gender_encoded = self.encode_label(gender, self.le_dict['department'])
        scent_encoded = self.encode_label(preferred_scent, self.le_dict['scents'])

        if gender_encoded == -1 or scent_encoded == -1:
            print(f"Warning: Unknown label detected. Gender: {gender}, Scent: {preferred_scent}")
            return []

        # 기분에 따른 향 인코딩
        mood_scents = self.mood_scent_mapping.get(mood, [])
        mood_encoded = [self.encode_label(scent, self.le_dict['scents']) for scent in mood_scents]

        # 상황에 따른 농도 인코딩
        situation_concentrations = self.situation_concentration_mapping.get(situation, [])
        situation_encoded = [self.encode_label(conc, self.le_dict['concentration']) for conc in situation_concentrations]

        # 사용자 벡터 생성 (성별, 선호 향기, 기분, 상황)
        mood_avg = np.mean(mood_encoded) if mood_encoded else 0
        situation_avg = np.mean(situation_encoded) if situation_encoded else 0
        user_vector = np.array([gender_encoded, scent_encoded, mood_avg, situation_avg])

        # 추천 필터: 사용자의 성별에 따라 필터링
        if gender == 'Men':
            valid_departments = [self.le_dict['department'].transform(['Men'])[0], self.le_dict['department'].transform(['Unisex'])[0]]
        elif gender == 'Women':
            valid_departments = [self.le_dict['department'].transform(['Women'])[0], self.le_dict['department'].transform(['Unisex'])[0]]
        else:
            valid_departments = self.le_dict['department'].classes_

        similarity_scores = []
        for name, group in self.data_exploded.groupby('name'):
            if group['department'].iloc[0] not in valid_departments:
                continue  # 성별에 맞지 않는 향수는 건너뜁니다.

            perfume_vector = np.array([group['department'].iloc[0], group['scents'].iloc[0], group['base_note_split'].mean(), group['concentration'].iloc[0]])
            similarity = self.calculate_similarity(user_vector, perfume_vector)

            # 예측된 평점 계산
            predicted_rating = self.rf_model.predict(perfume_vector.reshape(1, -1))[0]

            # 최종 점수 계산 (유사도와 예측 평점 결합)
            final_score = similarity * 0.7 + predicted_rating * 0.3

            # 평점 필터 적용
            if group['item_rating'].iloc[0] >= 3.5:
                similarity_scores.append((name, final_score, similarity, group.iloc[0]))

        # 상위 3개의 향수를 추천
        recommended_perfumes = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]

        return recommended_perfumes

    @pyqtSlot()
    def on_click_recommend(self):
        gender = self.gender_combobox.currentText()
        preferred_scent = self.scent_combobox.currentText()
        mood = self.mood_combobox.currentText()
        situation = self.situation_combobox.currentText()

        recommendations = self.recommend_perfume(gender, preferred_scent, mood, situation)

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
    app = QtWidgets.QApplication(sys.argv)
    ex = PerfumeRecommendationApp()
    ex.show()
    sys.exit(app.exec_())

