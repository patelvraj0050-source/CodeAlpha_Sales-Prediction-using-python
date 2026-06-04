import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style for high-quality visual insights
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Create a directory to save the output insights images
output_dir = "insights_outputs"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. LOAD AND CLEAN DATA
# ==========================================
print("--- Loading Dataset ---")
df = pd.read_csv("Advertising.csv")

# Drop named or unnamed index columns if present
if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
    df = df.iloc[:, 1:]

print(f"Dataset Shape: {df.shape}")
print("\nMissing Values:\n", df.isnull().sum())
print("\nBasic Summary Statistics:\n", df.describe())

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
# ==========================================
print("\n--- Generating Insight Images ---")

# Insight 1: Pairplot to check distributions and linear relationships
pair_plot = sns.pairplot(df, kind='reg', diag_kind='kde', 
                         plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.6}})
pair_plot.fig.suptitle("Feature Distributions and Regression Trends", y=1.02)
pair_plot.savefig(os.path.join(output_dir, "1_feature_relationships.png"), bbox_inches='tight')
plt.close()

# Insight 2: Correlation Heatmap
plt.figure(figsize=(8, 6))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", mask=mask, fmt=".3f", linewidths=0.5)
plt.title("Correlation Matrix of Advertising Channels vs Sales")
plt.savefig(os.path.join(output_dir, "2_correlation_heatmap.png"), bbox_inches='tight')
plt.close()

# Insight 3: Budget Allocations Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['TV', 'Radio', 'Newspaper']], palette="Set2")
plt.title("Distribution of Spending Across Advertising Channels")
plt.ylabel("Budget Spend ($ thousands)")
plt.savefig(os.path.join(output_dir, "3_budget_distributions.png"), bbox_inches='tight')
plt.close()

# ==========================================
# 3. DATA SPLITTING & PREPROCESSING
# ==========================================
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split: 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. MODEL TRAINING & HYPERPARAMETER TUNING
# ==========================================
print("\n--- Training Predictive Models ---")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Hyperparameter tuning for Random Forest to optimize performance
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, scoring='r2')
grid_rf.fit(X_train, y_train)
models["Tuned Random Forest"] = grid_rf.best_estimator_

# Evaluate all models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R² Score": r2}

results_df = pd.DataFrame(results).T
print("\nModel Evaluation Summary Table:")
print(results_df.to_string())

# ==========================================
# 5. POST-MODELING INSIGHTS & RESIDUAL ANALYSIS
# ==========================================
best_model_name = results_df['R² Score'].idxmax()
best_model = models[best_model_name]
best_preds = best_model.predict(X_test)

# Insight 4: Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=best_preds, color='purple', alpha=0.7, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f"Actual vs. Predicted Sales ({best_model_name})")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.savefig(os.path.join(output_dir, "4_actual_vs_predicted.png"), bbox_inches='tight')
plt.close()

# Insight 5: Feature Importance (Tree-Based Models)
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(8, 5))
    importances = best_model.feature_importances_
    sns.barplot(x=importances, y=X.columns, palette="viridis")
    plt.title(f"Feature Importance Weights - {best_model_name}")
    plt.xlabel("Relative Importance Score")
    plt.savefig(os.path.join(output_dir, "5_feature_importance.png"), bbox_inches='tight')
    plt.close()
elif hasattr(best_model, 'coef_'):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=best_model.coef_, y=X.columns, palette="coolwarm")
    plt.title(f"Model Coefficients - {best_model_name}")
    plt.xlabel("Coefficient Weight")
    plt.savefig(os.path.join(output_dir, "5_feature_importance.png"), bbox_inches='tight')
    plt.close()

print(f"\nExecution Complete! All plots saved successfully in the '{output_dir}' directory.")