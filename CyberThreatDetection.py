# corrected_cyber_threat_gui.py
from tkinter import messagebox, simpledialog, filedialog
import tkinter as tk
from tkinter import Label, Button, Text, Scrollbar, END
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing, svm, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pyswarms as ps

# Use tensorflow.keras consistently
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical

# -------------------------
# Globals (initialized)
# -------------------------
le = preprocessing.LabelEncoder()
filename = None
feature_extraction = None
X = None
Y = None
doc = None
label_names = None

X_train = X_test = y_train = y_test = None

# metrics (initialized to 0 so graphs won't crash)
lstm_acc = cnn_acc = svm_acc = knn_acc = dt_acc = random_acc = nb_acc = pso_accuracy = 0
lstm_precision = cnn_precision = svm_precision = knn_precision = dt_precision = random_precision = nb_precision = pso_precision = 0
lstm_recall = cnn_recall = svm_recall = knn_recall = dt_recall = random_recall = nb_recall = pso_recall = 0
lstm_fm = cnn_fm = svm_fm = knn_fm = dt_fm = random_fm = nb_fm = pso_fmeasure = 0

# classifier used in PSO fitness
classifier = linear_model.LogisticRegression(max_iter=1000)

# -------------------------
# Functions
# -------------------------
def upload():
    global filename, X, Y, doc, label_names
    filename = filedialog.askopenfilename(initialdir="datasets", title="Select CSV dataset")
    if not filename:
        return
    try:
        dataset = pd.read_csv(filename)
    except Exception as e:
        text.insert(END, f"Failed to load file: {e}\n")
        return

    if 'labels' not in dataset.columns:
        text.insert(END, "CSV must contain a 'labels' column.\n")
        return

    label_names = dataset['labels'].unique()
    dataset['labels'] = le.fit_transform(dataset['labels'])
    cols = dataset.shape[1] - 1
    X = dataset.values[:, 0:cols]
    Y = dataset.values[:, cols].astype(int)

    # Build a text doc representation if features are token-like or numeric features joined
    doc = []
    for i in range(len(X)):
        strs = ' '.join(str(v) for v in X[i])
        doc.append(strs)

    text.delete('1.0', END)
    text.insert(END, f"{os.path.basename(filename)} Loaded\n")
    text.insert(END, f"Total dataset size : {len(dataset)}\n")
    text.insert(END, f"Unique labels: {list(label_names)}\n")


def tfidf():
    global X, feature_extraction, doc
    if doc is None:
        text.insert(END, "Load dataset first.\n")
        return
    feature_extraction = TfidfVectorizer()
    tfidf_mat = feature_extraction.fit_transform(doc)
    X = tfidf_mat.toarray()
    text.delete('1.0', END)
    text.insert(END, 'TF-IDF processing completed\n')
    text.insert(END, f"Feature vector size: {X.shape}\n")


def eventVector():
    global X_train, X_test, y_train, y_test, X, Y, label_names
    if X is None or Y is None:
        text.insert(END, "Load dataset (and optionally run TF-IDF) first.\n")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    text.delete('1.0', END)
    text.insert(END, f"Total unique events found in dataset:\n{label_names}\n\n")
    text.insert(END, f"Total dataset size : {len(X)}\n")
    text.insert(END, f"Data used for training : {len(X_train)}\n")
    text.insert(END, f"Data used for testing  : {len(X_test)}\n")


def neuralNetwork():
    global lstm_acc, lstm_precision, lstm_recall, lstm_fm
    global cnn_acc, cnn_precision, cnn_recall, cnn_fm

    if X_train is None or X_test is None:
        text.insert(END, "Generate event vector (train/test split) first.\n")
        return

    # Prepare one-hot labels for neural networks
    num_classes = len(np.unique(Y))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    # ========= LSTM =========
    # LSTM expects 3D input (samples, timesteps, features). We'll treat feature vector as timesteps with 1 feature.
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train_lstm.shape[1], 1)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    text.insert(END, "Training LSTM (this may take time depending on data)...\n")
    hist = model.fit(X_train_lstm, y_train_cat, epochs=3, batch_size=128, verbose=0)
    pred = model.predict(X_test_lstm)
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(y_test_cat, axis=1)

    lstm_acc = accuracy_score(true_labels, pred_labels) * 100
    lstm_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0) * 100
    lstm_recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0) * 100
    lstm_fm = f1_score(true_labels, pred_labels, average='macro', zero_division=0) * 100

    text.insert(END, "\nDeep Learning LSTM Results\n")
    text.insert(END, f"LSTM Accuracy  : {lstm_acc:.2f}\n")
    text.insert(END, f"LSTM Precision : {lstm_precision:.2f}\n")
    text.insert(END, f"LSTM Recall    : {lstm_recall:.2f}\n")
    text.insert(END, f"LSTM Fmeasure  : {lstm_fm:.2f}\n")

    # ========= Simple Dense (CNN-style / feedforward) =========
    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(num_classes))
    cnn_model.add(Activation('softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    text.insert(END, "Training dense network (CNN-style)...\n")
    hist1 = cnn_model.fit(X_train, y_train_cat, epochs=10, batch_size=128, validation_split=0.2, shuffle=True, verbose=0)
    pred1 = cnn_model.predict(X_test)
    pred1_labels = np.argmax(pred1, axis=1)

    cnn_acc = accuracy_score(y_test, pred1_labels) * 100
    cnn_precision = precision_score(y_test, pred1_labels, average='macro', zero_division=0) * 100
    cnn_recall = recall_score(y_test, pred1_labels, average='macro', zero_division=0) * 100
    cnn_fm = f1_score(y_test, pred1_labels, average='macro', zero_division=0) * 100

    text.insert(END, "\nDeep Learning Dense Network Results\n")
    text.insert(END, f"CNN Accuracy  : {cnn_acc:.2f}\n")
    text.insert(END, f"CNN Precision : {cnn_precision:.2f}\n")
    text.insert(END, f"CNN Recall    : {cnn_recall:.2f}\n")
    text.insert(END, f"CNN Fmeasure  : {cnn_fm:.2f}\n")


def svmClassifier():
    global svm_acc, svm_precision, svm_recall, svm_fm
    if X_train is None:
        text.insert(END, "Generate event vector (train/test split) first.\n")
        return
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    svm_acc = accuracy_score(y_test, prediction_data) * 100
    svm_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    svm_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    svm_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100

    text.insert(END, "\nSVM Results\n")
    text.insert(END, f"SVM Precision : {svm_precision:.2f}\n")
    text.insert(END, f"SVM Recall : {svm_recall:.2f}\n")
    text.insert(END, f"SVM FMeasure : {svm_fm:.2f}\n")
    text.insert(END, f"SVM Accuracy : {svm_acc:.2f}\n")


def knn():
    global knn_acc, knn_precision, knn_recall, knn_fm
    if X_train is None:
        text.insert(END, "Generate event vector (train/test split) first.\n")
        return
    cls = KNeighborsClassifier(n_neighbors=10)
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    knn_acc = accuracy_score(y_test, prediction_data) * 100
    knn_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    knn_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    knn_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100

    text.insert(END, "\nKNN Results\n")
    text.insert(END, f"KNN Precision : {knn_precision:.2f}\n")
    text.insert(END, f"KNN Recall : {knn_recall:.2f}\n")
    text.insert(END, f"KNN FMeasure : {knn_fm:.2f}\n")
    text.insert(END, f"KNN Accuracy : {knn_acc:.2f}\n")


def randomForest():
    global random_acc, random_precision, random_recall, random_fm
    if X_train is None:
        text.insert(END, "Generate event vector (train/test split) first.\n")
        return
    cls = RandomForestClassifier(n_estimators=100, random_state=0)
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    random_acc = accuracy_score(y_test, prediction_data) * 100
    random_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    random_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    random_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100

    text.insert(END, "\nRandom Forest Results\n")
    text.insert(END, f"Random Forest Precision : {random_precision:.2f}\n")
    text.insert(END, f"Random Forest Recall : {random_recall:.2f}\n")
    text.insert(END, f"Random Forest FMeasure : {random_fm:.2f}\n")
    text.insert(END, f"Random Forest Accuracy : {random_acc:.2f}\n")


def naiveBayes():
    global nb_acc, nb_precision, nb_recall, nb_fm
    if X_train is None:
        text.insert(END, "Generate event vector (train/test split) first.\n")
        return
    cls = BernoulliNB(binarize=0.0)
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    nb_acc = accuracy_score(y_test, prediction_data) * 100
    nb_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    nb_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    nb_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100

    text.insert(END, "\nNaive Bayes Results\n")
    text.insert(END, f"Naive Bayes Precision : {nb_precision:.2f}\n")
    text.insert(END, f"Naive Bayes Recall : {nb_recall:.2f}\n")
    text.insert(END, f"Naive Bayes FMeasure : {nb_fm:.2f}\n")
    text.insert(END, f"Naive Bayes Accuracy : {nb_acc:.2f}\n")


# PSO helpers (fixed)
def f_per_particle(m, alpha):
    # m will be a binary mask vector for selected features
    global X, Y, classifier
    total_features = X.shape[1]
    mask = m.astype(bool)
    if mask.sum() == 0:
        X_subset = X
    else:
        X_subset = X[:, mask]
    # Use a simple validation split to estimate fitness (avoid training/predict on same data)
    X_tr, X_val, y_tr, y_val = train_test_split(X_subset, Y, test_size=0.3, random_state=42, stratify=Y)
    try:
        classifier.fit(X_tr, y_tr)
        P = (classifier.predict(X_val) == y_val).mean()
    except Exception:
        P = 0.0
    # objective: minimize j (we use same formula as original)
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / float(total_features))))
    return j


def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)


def SVMPSO():
    global pso_recall, pso_accuracy, pso_fmeasure, pso_precision, X, Y
    if X is None:
        text.insert(END, "Load dataset first.\n")
        return

    text.insert(END, f"\nTotal features in dataset before applying PSO : {X.shape[1]}\n")
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
    dimensions = X.shape[1]
    optimizer = ps.discrete.BinaryPSO(n_particles=6, dimensions=dimensions, options=options)
    # keep iterations small so it's faster; increase for better selection
    cost, pos = optimizer.optimize(f, iters=5)
    pos = pos.astype(bool)
    X_selected_features = X[:, pos]
    text.insert(END, f"Total features in dataset after applying PSO : {X_selected_features.shape[1]}\n")

    # train/test split on selected features
    X_tr, X_te, y_tr, y_te = train_test_split(X_selected_features, Y, test_size=0.2, random_state=42, stratify=Y)
    cls = svm.SVC()
    cls.fit(X_tr, y_tr)
    prediction_data = cls.predict(X_te)

    pso_accuracy = accuracy_score(y_te, prediction_data) * 100
    pso_precision = precision_score(y_te, prediction_data, average='macro', zero_division=0) * 100
    pso_recall = recall_score(y_te, prediction_data, average='macro', zero_division=0) * 100
    pso_fmeasure = f1_score(y_te, prediction_data, average='macro', zero_division=0) * 100

    text.insert(END, f"SVM with PSO Precision : {pso_precision:.2f}\n")
    text.insert(END, f"SVM with PSO Recall : {pso_recall:.2f}\n")
    text.insert(END, f"SVM with PSO FMeasure : {pso_fmeasure:.2f}\n")
    text.insert(END, f"SVM with PSO Accuracy : {pso_accuracy:.2f}\n")


def decisionTree():
    global dt_acc, dt_precision, dt_recall, dt_fm
    if X_train is None:
        text.insert(END, "Generate event vector (train/test split) first.\n")
        return
    cls = DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=3, min_samples_split=50, min_samples_leaf=20, max_features=5)
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    dt_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    dt_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    dt_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    dt_acc = accuracy_score(y_test, prediction_data) * 100

    text.insert(END, "\nDecision Tree Results\n")
    text.insert(END, f"Decision Tree Precision : {dt_precision:.2f}\n")
    text.insert(END, f"Decision Tree Recall : {dt_recall:.2f}\n")
    text.insert(END, f"Decision Tree FMeasure : {dt_fm:.2f}\n")
    text.insert(END, f"Decision Tree Accuracy : {dt_acc:.2f}\n")


def _safe_list(vals):
    # helper to ensure no NaN/None in lists for plotting
    return [0 if (v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))) else v for v in vals]


def graph():
    heights = _safe_list([knn_acc, nb_acc, dt_acc, svm_acc, random_acc, lstm_acc, cnn_acc, pso_accuracy])
    bars = ('KNN Acc', 'NB Acc', 'DT Acc', 'SVM Acc', 'RF Acc', 'LSTM Acc', 'CNN Acc', 'SVM PSO Acc')
    y_pos = np.arange(len(bars))
    plt.figure(figsize=(10, 4))
    plt.bar(y_pos, heights)
    plt.xticks(y_pos, bars, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def precisiongraph():
    heights = _safe_list([knn_precision, nb_precision, dt_precision, svm_precision, random_precision, lstm_precision, cnn_precision, pso_precision])
    bars = ('KNN Prec', 'NB Prec', 'DT Prec', 'SVM Prec', 'RF Prec', 'LSTM Prec', 'CNN Prec', 'PSO Prec')
    y_pos = np.arange(len(bars))
    plt.figure(figsize=(10, 4))
    plt.bar(y_pos, heights)
    plt.xticks(y_pos, bars, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def recallgraph():
    heights = _safe_list([knn_recall, nb_recall, dt_recall, svm_recall, random_recall, lstm_recall, cnn_recall, pso_recall])
    bars = ('KNN Rec', 'NB Rec', 'DT Rec', 'SVM Rec', 'RF Rec', 'LSTM Rec', 'CNN Rec', 'PSO Rec')
    y_pos = np.arange(len(bars))
    plt.figure(figsize=(10, 4))
    plt.bar(y_pos, heights)
    plt.xticks(y_pos, bars, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def fmeasuregraph():
    heights = _safe_list([knn_fm, nb_fm, dt_fm, svm_fm, random_fm, lstm_fm, cnn_fm, pso_fmeasure])
    bars = ('KNN F', 'NB F', 'DT F', 'SVM F', 'RF F', 'LSTM F', 'CNN F', 'PSO F')
    y_pos = np.arange(len(bars))
    plt.figure(figsize=(10, 4))
    plt.bar(y_pos, heights)
    plt.xticks(y_pos, bars, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# -------------------------
# GUI Layout (kept similar)
# -------------------------
main = tk.Tk()
main.title("Cyber Threat Detection Based on Artificial Neural Networks Using Event Profiles")
main.geometry("1300x800")

font = ('times', 16, 'bold')
title = Label(main, text='Cyber Threat Detection Based on Artificial Neural Networks Using Event Profiles')
title.config(bg='darkviolet', fg='gold')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
text.config(font=font1)
text.place(x=50, y=120)

scroll = Scrollbar(main, command=text.yview)
text.configure(yscrollcommand=scroll.set)
scroll.place(x=1230, y=120, height=330)

uploadButton = Button(main, text="Upload Train Dataset", command=upload)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Run Preprocessing TF-IDF Algorithm", command=tfidf)
preprocessButton.place(x=240, y=550)
preprocessButton.config(font=font1)

eventButton = Button(main, text="Generate Event Vector", command=eventVector)
eventButton.place(x=535, y=550)
eventButton.config(font=font1)

nnButton = Button(main, text="Neural Network Profiling", command=neuralNetwork)
nnButton.place(x=730, y=550)
nnButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=svmClassifier)
svmButton.place(x=950, y=550)
svmButton.config(font=font1)

knnButton = Button(main, text="Run KNN Algorithm", command=knn)
knnButton.place(x=1130, y=550)
knnButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest Algorithm", command=randomForest)
rfButton.place(x=50, y=600)
rfButton.config(font=font1)

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=naiveBayes)
nbButton.place(x=320, y=600)
nbButton.config(font=font1)

dtButton = Button(main, text="Run Decision Tree Algorithm", command=decisionTree)
dtButton.place(x=570, y=600)
dtButton.config(font=font1)

psoButton = Button(main, text="Extension SVM with PSO", command=SVMPSO)
psoButton.place(x=830, y=600)
psoButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=1080, y=600)
graphButton.config(font=font1)

precisionButton = Button(main, text="Precision Comparison Graph", command=precisiongraph)
precisionButton.place(x=50, y=650)
precisionButton.config(font=font1)

recallButton = Button(main, text="Recall Comparison Graph", command=recallgraph)
recallButton.place(x=320, y=650)
recallButton.config(font=font1)

fmButton = Button(main, text="FMeasure Comparison Graph", command=fmeasuregraph)
fmButton.place(x=570, y=650)
fmButton.config(font=font1)

main.config(bg='turquoise')

main.mainloop()
