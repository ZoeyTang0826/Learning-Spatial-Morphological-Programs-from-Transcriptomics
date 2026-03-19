import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/path/to/test_feature_metrics_delta.csv")

df_sorted = df.sort_values("delta_r2_conditional", ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(df_sorted["feature"], df_sorted["delta_r2_conditional"])
plt.axvline(0, linestyle="--")
plt.title("Feature-level Improvement (ΔR²)")
plt.xlabel("ΔR² (conditional - base)")
plt.gca().invert_yaxis()
plt.savefig("/path/to/feature_importance_delta_r2.png", bbox_inches="tight")