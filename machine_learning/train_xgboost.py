import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# 기존 전처리 함수 불러오기
from preprocessor import load_and_preprocess

# 1) train/test CSV 경로 설정
base_dir    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
train_path  = os.path.join(base_dir, "result.csv")
test_path   = os.path.join(base_dir, "test.csv")

# 2) 각각 불러와서 X, y 획득
print(f"[INFO] Train CSV 로딩: {train_path}")
X_train, y_train = load_and_preprocess(train_path)
print(f"[INFO]   → {len(y_train)} samples")

print(f"[INFO] Test  CSV 로딩: {test_path}")
X_test, y_test = load_and_preprocess(test_path)
print(f"[INFO]   → {len(y_test)} samples")

# 3) 모델 학습
print("[INFO] XGBoost 학습 시작...")
model = XGBClassifier(
    n_estimators=100,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

# 4) 테스트셋 평가 (threshold 적용)
print("\n[RESULT] 테스트셋 평가 (threshold 적용)")
probs = model.predict_proba(X_test)[:, 1]  # 클래스 1의 확률
threshold = 0.718
y_pred = (probs >= threshold).astype(int)
print(f">>> Using threshold = {threshold}\n")
print(classification_report(y_test, y_pred, zero_division=0))

# 5) 추가 지표 (multiclass macro)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

# ROC-AUC (binary)
roc_auc = roc_auc_score(y_test, probs)

# PR-AUC (binary)
pr_auc_bin = average_precision_score(y_test, probs)

# multiclass PR-AUC: One-vs-Rest 방식
probs_full = model.predict_proba(X_test)
pr_list = [
    average_precision_score((y_test == i).astype(int), probs_full[:, i])
    for i in range(probs_full.shape[1])
]
pr_auc_macro = sum(pr_list) / len(pr_list)

# 클래스별 recall gap
rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
recalls = [rep[k]["recall"] for k in rep if k.isdigit()]
recall_gap = max(recalls) - min(recalls) if recalls else 0.0

# 결과 출력
print(f"\n📊 Precision (macro) : {precision:.2f}")
print(f"📊 Recall    (macro) : {recall:.2f}")
print(f"📊 F1 Score  (macro) : {f1_macro:.2f}")
print(f"📊 ROC-AUC  (binary) : {roc_auc:.2f}")
print(f"📊 PR-AUC   (binary) : {pr_auc_bin:.2f}")
print(f"📊 PR-AUC   (macro)  : {pr_auc_macro:.2f}")
print(f"📊 Recall Gap        : {recall_gap:.2f}")

# 5-1) 혼동 행렬 시각화
print("\n[INFO] 혼동 행렬 시각화")
cm_display = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["normal (0)", "detected (1)"],
    cmap=plt.cm.Blues,
    values_format='d'
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 6) 모델 저장
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))
os.makedirs(out_dir, exist_ok=True)
save_path = os.path.join(out_dir, "xgb_model.pkl")
joblib.dump(model, save_path)
print(f"\n✅ 모델 저장 완료: {save_path}")
