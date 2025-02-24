import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 가상의 데이터 준비
data = {
    'emails_sent': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    'click_rate': [1.2, 2.5, 3.8, 4.21, 5.1, 6.3, 7.0, 7.8, 8.5, 9.2]
}
df = pd.DataFrame(data)

# 2. 독립 변수(X)와 종속 변수(y) 설정
X = df[['emails_sent']]
y = df['click_rate']

# 3. 단순 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 4. 모델 파라미터 및 결정계수(R²) 출력
intercept = model.intercept_
slope = model.coef_[0]
r2 = r2_score(y, model.predict(X))

print("회귀식: 클릭률 = {:.2f} + {:.5f} * 발송한 이메일 수".format(intercept, slope))
print("결정계수 (R²): {:.4f}".format(r2))

# 5. 예측 및 시각화
df['predicted_clicks'] = model.predict(X)

plt.scatter(df['emails_sent'], df['click_rate'], color='blue', label='실제 클릭률')
plt.plot(df['emails_sent'], df['predicted_clicks'], color='red', label='회귀 직선')
plt.title('발송 이메일 수 대비 클릭률 분석')
plt.xlabel('이메일 수')
plt.ylabel('클릭률')
plt.legend()
plt.show()
