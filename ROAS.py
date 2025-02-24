import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 가상의 데이터 준비
data = {
    'ad_spend': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    'conversions': [120, 150, 180, 210, 240, 265, 290, 320, 345, 370]
}
df = pd.DataFrame(data)

# 2. 독립 변수(X)와 종속 변수(y) 설정
X = df[['ad_spend']]
y = df['conversions']

# 3. 단순 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 4. 모델 파라미터 및 결정계수(R²) 출력
intercept = model.intercept_
slope = model.coef_[0]
r2 = r2_score(y, model.predict(X))

print("회귀식: 전환수 = {:.2f} + {:.5f} * 광고비".format(intercept, slope))
print("결정계수 (R²): {:.4f}".format(r2))

# 5. 예측 및 시각화
df['predicted_conversions'] = model.predict(X)

plt.scatter(df['ad_spend'], df['conversions'], color='blue', label='실제 전환수')
plt.plot(df['ad_spend'], df['predicted_conversions'], color='red', label='회귀 직선')
plt.title('광고비 대비 전환수 분석')
plt.xlabel('광고비 ($)')
plt.ylabel('전환수')
plt.legend()
plt.show()
