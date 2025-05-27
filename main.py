import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import shap
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    filename='housing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



class DataLoader:
    """稳健数据加载器"""

    def __init__(self, url):
        self.url = url
        self.df = None

    def load_data(self):
        """加载并验证数据"""
        try:
            self.df = pd.read_csv(self.url)
            self._clean_data()
            logging.info(f"数据加载成功，维度：{self.df.shape}")
            return self.df
        except Exception as e:
            logging.error(f"数据加载失败：{str(e)}")
            raise

    def _clean_data(self):
        """数据清洗"""
        # 处理缺失值
        self.df = self.df.dropna(subset=['median_income', 'housing_median_age'])

        # 处理极端值
        self.df['median_house_value'] = self.df['median_house_value'].clip(upper=500000)

        # 确保数值合法性
        self.df['total_rooms'] = self.df['total_rooms'].replace(0, 1)
        self.df['households'] = self.df['households'].replace(0, 1)



class FeatureEngineer(BaseEstimator, TransformerMixin):
    """可复用的特征工程管道"""

    def __init__(self):
        self.feature_names_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """生成新特征"""
        df = X.copy()

        # 空间特征（对数变换）
        df['rooms_per_household'] = np.log1p(df['total_rooms'] / df['households'])
        df['bedrooms_ratio'] = np.log1p(df['total_bedrooms'] / df['total_rooms'])

        # 时间特征
        df['house_age'] = 2023 - df['housing_median_age']

        # 地理特征
        df['distance_to_coast'] = np.sqrt(
            (df['latitude'] - 34.42) ** 2 +
                                         (df['longitude'] + 118.49) ** 2
        )

        # 选择特征
        self.feature_names_ = [
            'median_income', 'house_age', 'rooms_per_household',
            'bedrooms_ratio', 'distance_to_coast', 'ocean_proximity'
        ]
        return df[self.feature_names_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_



class SafeOneHotEncoder(BaseEstimator, TransformerMixin):
    """稳健的类别编码器"""

    def __init__(self):
        self.categories_ = {}
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        # 记录所有类别
        self.categories_ = {
            col: X_df[col].unique().tolist()
            for col in X_df.columns
        }
        # 生成特征名称
        self.feature_names_out_ = [
            f"{col}_{cat}"
            for col in X_df.columns
            for cat in sorted(self.categories_[col])
        ]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        # 生成独热编码
        dummies = pd.get_dummies(X_df, prefix_sep='_')
        # 对齐特征
        for col in self.feature_names_out_:
            if col not in dummies.columns:
                dummies[col] = 0
        return dummies[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        # 明确返回列表类型
        return self.feature_names_out_  # 直接返回列表


def build_preprocessor():
    """构建预处理管道"""
    num_features = ['median_income', 'house_age',
                    'rooms_per_household', 'bedrooms_ratio',
                    'distance_to_coast']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('onehot', SafeOneHotEncoder())
    ])

    return ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, ['ocean_proximity'])
    ])



def train_model(X_train, y_train):
    """模型训练与调优"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    logging.info(f"最佳参数：{grid_search.best_params_}")
    return grid_search.best_estimator_



def visualize_results(model, X_test, feature_names):
    """生成SHAP解释图"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')



def main():
    try:
        # 数据加载
        loader = DataLoader("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv")
        df = loader.load_data()

        # 特征工程
        engineer = FeatureEngineer()
        X = engineer.fit_transform(df)
        y = df['median_house_value']

        # 数据拆分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 预处理
        preprocessor = build_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # 正确获取特征名称
        num_features = ['median_income', 'house_age',
                        'rooms_per_household', 'bedrooms_ratio',
                        'distance_to_coast']  # 直接使用列表

        cat_features = preprocessor.named_transformers_['cat'].named_steps[
            'onehot'].get_feature_names_out()  # 已经是列表

        all_features = num_features + cat_features  # 直接拼接
        # 验证特征类型
        print(f"数值特征类型: {type(num_features)}")  # <class 'list'>
        print(f"类别特征类型: {type(cat_features)}")  # <class 'list'>
        print(f"总特征类型: {type(all_features)}")  # <class 'list'>

        # 模型训练
        model = train_model(X_train_processed, y_train)

        # 模型评估
        y_pred = model.predict(X_test_processed)
        print("\n=== 模型性能 ===")
        print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.0f}")
        print(f"R²: {r2_score(y_test, y_pred):.2f}")

        # 可视化
        visualize_results(model, X_test_processed, all_features)
        # 保存模型
        joblib.dump(model, 'final_model.pkl')
        joblib.dump(preprocessor, 'preprocessor.pkl')
        logging.info("模型保存成功")

    except Exception as e:
        logging.error(f"系统错误：{str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()