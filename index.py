import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('./data/Dados_RH_Turnover.csv', delimiter=';')

df = pd.get_dummies(df, columns=['DeptoAtuacao', 'Salario'], drop_first=True)

X = df.drop('SaiuDaEmpresa', axis=1)
y = df['SaiuDaEmpresa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Decision Tree': DecisionTreeClassifier(),
    'K-NN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Neural Network': MLPClassifier(max_iter=300)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'Accuracy': accuracy,
        'Confusion Matrix': matrix
    }

for model_name, metrics in results.items():
    print(f"{model_name} - Accuracy: {metrics['Accuracy']}")
    print(f"{model_name} - Confusion Matrix:\n{metrics['Confusion Matrix']}\n")

for model_name, metrics in results.items():
    accuracy = metrics['Accuracy']
    tn, fp, fn, tp = metrics['Confusion Matrix'].ravel()
    
    print(f"\nModelo: {model_name}")
    print(f"- Acurácia: {accuracy:.2%}")
    print(f"- Verdadeiros Negativos (TN): {tn} - O modelo corretamente previu que esses funcionários NÃO sairiam.")
    print(f"- Falsos Positivos (FP): {fp} - O modelo previu que esses funcionários sairiam, mas eles ficaram.")
    print(f"- Falsos Negativos (FN): {fn} - O modelo previu que esses funcionários ficariam, mas eles saíram.")
    print(f"- Verdadeiros Positivos (TP): {tp} - O modelo corretamente previu que esses funcionários sairiam.\n")
    
    # Resumo da performance do modelo
    print(f"Resumo do {model_name}:")
    if accuracy >= 0.9:
        print(f"O modelo {model_name} possui alta precisão e é adequado para prever a saída de funcionários.")
    elif 0.75 <= accuracy < 0.9:
        print(f"O modelo {model_name} tem um desempenho moderado. Ele pode ser útil, mas precisa de melhorias.")
    else:
        print(f"O modelo {model_name} apresenta baixa precisão e talvez não seja adequado para previsões confiáveis.\n")