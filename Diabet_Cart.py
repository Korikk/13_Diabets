import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier
from helpers.helpers import *
from helpers.data_prep import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_pickle("Datasets/prepared_diabetes_df.pkl")
check_df(df)

y = df["Outcome"]
x = df.drop("Outcome", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

x_train.shape
y_train.shape

cls = DecisionTreeClassifier(random_state=17)

model = cls.fit(x_train, y_train)
proba = model.predict_proba(x_test)[:, 1]
roc_auc_score(y_test, proba)


def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(model, x_train)

################################
cart_model = DecisionTreeClassifier(random_state=17)

# hyperparameters
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": [1, 2, 3, 4]}

cart_cv = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=True)
cart_cv.fit(x_train, y_train)

cart_cv.best_params_

cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_).fit(x_train, y_train)


y_pred = cart_tuned.predict(x_test)
y_prob = cart_tuned.predict_proba(x_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)

# 0.81

plot_importance(cart_tuned, x_train)