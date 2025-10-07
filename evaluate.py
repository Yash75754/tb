# src/evaluate.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

MODEL_DIR = "models/best_model"
model = tf.keras.models.load_model(MODEL_DIR)

# Build test dataset same as training
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/processed/test", image_size=(224,224), batch_size=16, label_mode="binary"
)
# get predictions and true labels
y_true = np.concatenate([y.numpy() for x,y in test_ds], axis=0)
y_pred_proba = np.concatenate([model.predict(x).ravel() for x,y in test_ds])
y_pred = (y_pred_proba >= 0.5).astype(int)

print(classification_report(y_true, y_pred, target_names=["NORMAL","TB"]))
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
print("ROC saved as roc_curve.png")
