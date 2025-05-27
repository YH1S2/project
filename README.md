# California Housing Price Prediction

## 🚀 Usage

Run the main script:
```bash
python main.py
```

​**Expected outputs**:
- `final_model.pkl` - Serialized trained model
- `preprocessor.pkl` - Fitted preprocessing pipeline
- `housing.log` - Execution logs
- `feature_importance.png` - SHAP feature importance visualization

## 📂 File Structure

| File                        | Description                          |
|-----------------------------|--------------------------------------|
| `main.py`                   | Main implementation pipeline         |
| `final_model.pkl`           | Trained Random Forest model          |
| `preprocessor.pkl`          | Fitted preprocessing transformer     |
| `housing.log`               | Runtime execution logs               |
| `feature_importance.png`    | SHAP feature importance diagram      |

## 🛠 Implementation Details

### Data Pipeline
```python
# Full preprocessing flow
DataLoader(url).load_data() 
→ FeatureEngineer() 
→ ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', SafeOneHotEncoder(), ['ocean_proximity'])
])
```

### Model Configuration
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
```
- ​**Validation**: 5-fold cross-validation
- ​**Metrics**: MAE (Mean Absolute Error), R² Score

## 📊 Results

| Metric        | Training Set      | Test Set          |
|---------------|-------------------|-------------------|
| MAE           | $38,421           | $43,217           |
| R² Score      | 0.82              | 0.75              |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch:
```bash
git checkout -b feature/your-feature
```
3. Commit changes:
```bash
git commit -m 'Add meaningful message'
```
4. Push branch:
```bash
git push origin feature/your-feature
``` 
5. Open a Pull Request

---

> ​**Note**: Requires Python 3.8+ and internet connection for initial data download. Full logs are recorded in `housing.log`.
