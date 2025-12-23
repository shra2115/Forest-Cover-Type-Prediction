# Forest-Cover-Type-Prediction

**Data Preparation and Machine Learning Setup**
This code sets up the environment for a complete machine learning workflow. It imports essential libraries for data handling, visualization, preprocessing, and modeling. The dataset train.csv is loaded into a DataFrame and displayed for initial inspection. Tools for scaling, encoding, dimensionality reduction, and handling class imbalance are included. Various classification models and evaluation metrics are also prepared for building and assessing predictive models.

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/content/train.csv')

df.head()


**Dataset Overview**

The df.info() shows the dataset’s structure, column types, and missing values. df.describe() provides statistical summaries like mean, min, max, and quartiles for numerical features. Together, they give a quick understanding of data quality and distribution. This helps identify issues and guides preprocessing. It is an essential first step in data analysis.

df.info()

df.describe()


**One-Hot Encoding to Single Label Conversion**

This function `onehot_to_label_idxmax` converts multiple one-hot encoded columns for wilderness areas into a single categorical column. It selects the column with the maximum value for each row using `idxmax()` and extracts the area number. The original one-hot columns are then dropped to simplify the dataset. This reduces dimensionality and makes the data easier to work with for machine learning models. Finally, the transformed DataFrame replaces the original one.

def onehot_to_label_idxmax(df):

	df = df.copy()
	
	cols = ['Wilderness_Area1', 'Wilderness_Area2',
	
			'Wilderness_Area3', 'Wilderness_Area4']
  
    df['Wilderness_Area'] = df[cols].idxmax(axis=1).str.replace('Wilderness_Area', '')
    
	return df.drop(columns=cols)

df = onehot_to_label_idxmax(df)

def onehot_to_label_idxmax(df):
    

	df = df.copy()

	cols = [f"Soil_Type{i}" for i in range(1, 41)]

    df["Soil_Type"] = df[cols].idxmax(axis=1).str.replace("Soil_Type", "")
    
	return df.drop(columns=cols)

df = onehot_to_label_idxmax(df)

**Soil Type Column Conversion**

This function converts 40 one-hot encoded soil type columns into a single column. It finds the active soil type in each row and keeps only that as a label. The original 40 columns are dropped to simplify the dataset. This makes the data easier to use for machine learning models. The updated DataFrame replaces the original one.


le = LabelEncoder()

df['Wilderness_Area'] = le.fit_transform(df['Wilderness_Area'])

df['Soil_Type'] = le.fit_transform(df['Soil_Type'])


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Cover_Type'])

print("Training set shape:", df_train.shape)

print("Testing set shape:", df_test.shape)

numeric_df = df_train

numeric_df.hist(bins=20, figsize=(15, 10), layout=(-1, 3), edgecolor='black')

plt.suptitle('Distribution of All Numeric Columns (Pandas Method)', y=1.02, fontsize=16)

plt.tight_layout()

plt.show()





import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

numeric_cols = df_train.columns.difference(['Cover_Type', 'Wilderness_Area', 'Soil'])

for w in sorted(df_train['Wilderness_Area'].unique()):

	subset = df_train[df_train['Wilderness_Area'] == w]
    
	if subset.empty:
    
		continue

    melted = subset.melt(value_vars=numeric_cols,
      
						  var_name='Metric',
                          
						  value_name='Value')

    plt.figure(figsize=(20, 8))
   
	sns.boxplot(data=melted, x='Metric', y='Value')
    
	plt.title(f'Distribution of Metrics in Wilderness Area "{w}"', fontsize=16, pad=20)
    
	plt.xlabel('Metric')
    
	plt.ylabel('Value')
    
	plt.xticks(rotation=45, ha='right')
    
	plt.grid(axis='y', linestyle='--', alpha=0.7)
    
	plt.tight_layout()
    
	plt.show()

    sns.heatmap(df_train.drop(columns=['Cover_Type',"Soil_Type","Wilderness_Area"]).corr(), annot=True, cmap='coolwarm')

plt.title('Correlation Heatmap')

plt.show()

Step 7:

sns.heatmap(df_train.drop(columns=['Cover_Type',"Soil_Type","Wilderness_Area"]).corr(), annot=True, cmap='coolwarm')

plt.title('Correlation Heatmap')

plt.show()

Step 8:

cover_type_counts = df_train['Cover_Type'].value_counts()

plt.figure(figsize=(8, 8))

plt.pie(cover_type_counts, labels=cover_type_counts.index, autopct='%1.1f%%', startangle=140)

plt.title('Distribution of Cover Types')

plt.axis('equal')

plt.show()

sns.pairplot(df_train.drop(columns=['Cover_Type',"Wilderness_Area","Soil_Type"]))

plt.suptitle('Pairwise Scatter Plots of Training Data Features', y=1.02, fontsize=16)

plt.tight_layout()

plt.show()

Step 9:


from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

X = df_train.drop('Cover_Type', axis=1)

y = df_train['Cover_Type']

X_pca = PCA(n_components=3).fit_transform(

	StandardScaler().fit_transform(X)
)


fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')

for i in range(1, 8):

	ax.scatter(*X_pca[y == i].T, label=f'Cover Type {i}')

ax.set(xlabel='Principal Component 1',
    
	   ylabel='Principal Component 2',
       
	   zlabel='Principal Component 3',
       
	   title='3D PCA of Cover Type Data')

ax.legend()

plt.show()

df.isnull().sum()

df['Direct_Distance_To_Hydrology'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)

df = df.drop(["Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology"],axis=1)

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import joblib

cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

df['Hillshade_PCA'] = PCA(n_components=1).fit_transform(

	StandardScaler().fit_transform(df[cols])
)

joblib.dump(StandardScaler().fit(df[cols]), 'scaler.pkl')

joblib.dump(PCA(n_components=1).fit(StandardScaler().fit_transform(df[cols])), 'pca.pkl')

print("Scaler and PCA objects saved.")

print(df[cols + ['Hillshade_PCA']].head())

df.drop(columns=cols, inplace=True)

X = df.drop('Cover_Type', axis=1)

y = df['Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(

	X, y,
    
	test_size=0.2,
    
	random_state=42
)

feature_columns = [col for col in df.columns if col != 'Cover_Type']

target_column = 'Cover_Type'

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

desired_sample_size = 100000

actual_train_size = len(X_train)

sample_size = min(desired_sample_size, actual_train_size)

sample_indices = X_train.sample(

	n=sample_size,
    
	random_state=42,
    
	replace=False

).index

X_train_sampled = X_train.loc[sample_indices]

y_train_sampled = y_train.loc[sample_indices]

print(f"Actual training size: {actual_train_size}")

print(f"Used sample size: {sample_size}")

print(f"Shape of X_train_sampled: {X_train_sampled.shape}")

print(f"Shape of y_train_sampled: {y_train_sampled.shape}")

etc_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

etc_model.fit(X_train, y_train)

y_pred_etc = etc_model.predict(X_test)

print("📋 Classification Report (Test Set):")

print(classification_report(y_test, y_pred_etc))

print("📋 Classification Report (Test Set):")

report = classification_report(y_test, y_pred_etc, target_names=[str(c) for c in sorted(y_test.unique())])

print(report

cm = confusion_matrix(y_test, y_pred_etc)

plt.figure(figsize=(10, 7))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y_test.unique()),
            yticklabels=sorted(y_test.unique()))

plt.title('Confusion Matrix for Tuned Extra Trees Classifier', fontsize=16, pad=20)

plt.xlabel('Predicted Label', fontsize=12)

plt.ylabel('Actual Label', fontsize=12)

plt.show()

from sklearn.ensemble import ExtraTreesClassifier

best_etc = ExtraTreesClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

best_etc.fit(X_train_sampled, y_train_sampled

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


importances_etc = best_etc.feature_importances_

feature_importance_df_etc = pd.DataFrame({
    'Feature': X_train_sampled.columns,
    'Importance': importances_etc
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))

sns.barplot(x='Importance', y='Feature', data=feature_importance_df_etc, palette='viridis')

plt.title('Feature Importance from Tuned Extra Trees Classifier', fontsize=16, pad=20)

plt.xlabel('Importance Score', fontsize=12)

plt.ylabel('Features', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
