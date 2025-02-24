```python     
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
!pip install koreanize-matplotlib
import koreanize_matplotlib
import pandas as pd

data = {
    "snsads_spend": [800, 1600, 2400, 3200, 4000, 4800, 5600, 6400, 7200, 8000],
    "influencer_budget": [500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300],
    "attend": [2.1, 2.5, 3.0, 3.3, 3.9, 4.1, 4.5, 4.8, 5.2, 5.6],
    "web_vistiors": [3000, 4500, 6000, 7500, 9000, 10500, 12000, 13500, 15000, 16500],
    "click_ads": [500, 900, 1400, 1900, 2500, 3100, 3700, 4300, 4900, 5500],
    "share_sns": [50, 80, 120, 160, 210, 270, 330, 400, 480, 550],
    "sales": [12000, 17000, 23000, 28000, 34000, 39000, 45000, 50000, 56000, 62000]
}

df = pd.DataFrame(data)

# 2. 독립 변수와 종속 변수 설정
X = df[['snsads_spend', 'web_vistiors', 'share_sns']]
y = df['sales']

# 3. 다중 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 4. 모델 파라미터 및 결정계수(R²) 출력
print("회귀식: 매출 = {:.2f} + {:.5f} * sns 광고비 + {:.5f} * 웹사이트 방문자 수 + {:.5f} * sns 공유 수".format(
    model.intercept_, model.coef_[0], model.coef_[1], model.coef_[2]
))
print("결정계수 (R²): {:.4f}".format(r2_score(y, model.predict(X))))

#	5. sns 광고비: 9000달러, 웹페이지 방문자 수: 20000명, sns 공유수: 600회의 추가조건이 주어졌을 때, 예상 매출 및 ROI 평가
pred_sales = model.intercept_ + model.coef_[0]*9000 +model.coef_[1]*20000 + model.coef_[2]*600
print(pred_sales)
roi =((pred_sales - 9000 )/ 9000 ) * 100
print(roi)


# 5. 실제 매출과 예측 매출 비교 산점도 시각화
predicted_sales = model.predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(y, predicted_sales, color='blue', label='예측 매출')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='대각선 (y=x)')
plt.xlabel('실제 매출 (달러)')
plt.ylabel('예측 매출 (달러)')
plt.title('실제 매출 vs 예측 매출')
plt.legend()
plt.show()
```
