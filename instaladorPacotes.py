import subprocess
import sys

required_packages = [
    'pandas', 
    'scikit-learn'
]

def install_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages(required_packages)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

print("Todos os pacotes necessários foram instalados e importados!")