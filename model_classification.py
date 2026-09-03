# ============================================
# 1. IMPORT LIBRARY
# ============================================
import pandas as pd
import numpy as np
import os
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


# ============================================
# 2. FUNCTION: PERSIAPAN DATA (FIT DI TRAIN, TRANSFORM KE TEST)
# ============================================
def prepare_data(df_train, df_test, target_col, scale=True):
    """
    Fungsi untuk memisahkan fitur & target dari data train dan test yang SUDAH terpisah.
    Encoding & scaling di-fit HANYA di data train, lalu ditransformasikan ke data test
    (supaya tidak ada data leakage dari test ke train).

    Parameters:
    - df_train: pandas DataFrame, dataset training
    - df_test: pandas DataFrame, dataset testing
    - target_col: str, nama kolom target/label
    - scale: bool, apakah fitur numerik perlu di-scaling

    Returns:
    - X_train, X_test, y_train, y_test
    """
    # Pisahkan fitur dan target
    X_train = df_train.drop(columns=[target_col]).copy()
    y_train = df_train[target_col].copy()

    X_test = df_test.drop(columns=[target_col]).copy()
    y_test = df_test[target_col].copy()

    # Deteksi kolom kategorikal dan numerik (berdasarkan data train)
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Kolom kategorikal terdeteksi: {cat_cols}")
    print(f"Kolom numerik terdeteksi: {num_cols}\n")

    # ---- One-Hot Encoding (fit kolomnya dari train, lalu samakan ke test) ----
    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

        # Samakan kolom hasil one-hot antara train & test
        # (kalau ada kategori yang cuma muncul di salah satu, isi 0 di yang lain)
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # ---- Encode target kalau masih string/bool (fit dari train) ----
    if y_train.dtype == 'object' or str(y_train.dtype) == 'category' or y_train.dtype == 'bool':
        y_train_cat = y_train.astype('category')
        categories = y_train_cat.cat.categories

        y_train = y_train_cat.cat.codes
        y_test = pd.Categorical(y_test, categories=categories).codes

    # ---- Scaling (fit HANYA di train, transform ke train & test) ----
    if scale and num_cols:
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test


# ============================================
# 3. FUNCTION: TRAINING MODEL (PER ALGORITMA)
# ============================================
def train_logistic_regression(X_train, y_train):
    """Melatih model Logistic Regression dengan hyperparameter default."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    """Melatih model SVM (SVC) dengan hyperparameter default."""
    model = SVC()
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Melatih model XGBoost dengan hyperparameter default."""
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


# ============================================
# 4. FUNCTION: HITUNG METRIK (DIPAKAI BARENG OLEH VALIDASI & TEST)
# ============================================
def compute_metrics(model, X, y):
    """
    Menghitung accuracy, precision, recall, f1, dan ROC-AUC untuk satu model
    pada satu set data (bisa data validasi/fold, bisa data test).
    Dipakai bersama oleh cross_validate_model() dan evaluate_model() supaya
    cara hitungnya konsisten di kedua tempat.

    Parameters:
    - model: model yang SUDAH di-fit
    - X, y: data yang mau dievaluasi

    Returns:
    - dict berisi metrik: accuracy, precision, recall, f1_score, roc_auc
    """
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

    # ---- Hitung ROC-AUC ----
    n_classes = len(np.unique(y))

    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X)
        else:
            y_score = None

        if y_score is not None:
            if n_classes == 2:
                # Untuk binary classification, ambil probabilitas kelas positif
                if y_score.ndim > 1:
                    y_score_bin = y_score[:, 1]
                else:
                    y_score_bin = y_score
                roc_auc = roc_auc_score(y, y_score_bin)
            else:
                # Untuk multiclass, pakai strategi One-vs-Rest
                roc_auc = roc_auc_score(y, y_score, multi_class='ovr', average='weighted')
        else:
            roc_auc = np.nan
    except Exception:
        roc_auc = np.nan

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


# ============================================
# 5. FUNCTION: CROSS-VALIDATION (5-FOLD, DARI DATA TRAIN)
# ============================================
def cross_validate_model(model, X_train, y_train, cv=5):
    """
    Melakukan cross-validation manual pada data training, menghitung metrik
    LENGKAP (accuracy, precision, recall, f1, roc_auc) di setiap fold,
    lalu merata-ratakannya. Ini dianggap sebagai hasil "VALIDASI", terpisah
    dari hasil evaluasi di data test.

    Parameters:
    - model: model (belum di-fit) — akan di-clone tiap fold supaya tidak
      saling mempengaruhi antar fold
    - X_train, y_train: data training
    - cv: jumlah fold (default 5)

    Returns:
    - dict berisi rata-rata (mean) dan std tiap metrik di seluruh fold
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_metrics = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model_fold = clone(model)
        model_fold.fit(X_tr, y_tr)

        metrics = compute_metrics(model_fold, X_val, y_val)
        fold_metrics.append(metrics)

        print(f"  Fold {fold_idx}: "
              f"acc={metrics['accuracy']:.4f}, "
              f"prec={metrics['precision']:.4f}, "
              f"rec={metrics['recall']:.4f}, "
              f"f1={metrics['f1_score']:.4f}, "
              f"roc_auc={metrics['roc_auc']:.4f}" if not np.isnan(metrics['roc_auc'])
              else f"  Fold {fold_idx}: acc={metrics['accuracy']:.4f}, roc_auc=NaN")

    fold_df = pd.DataFrame(fold_metrics)
    result = {}
    for col in fold_df.columns:
        result[f"{col}_mean"] = fold_df[col].mean()
        result[f"{col}_std"] = fold_df[col].std()

    print(f"\nRata-rata {cv}-fold Cross-Validation:")
    for col in fold_df.columns:
        print(f"  {col:<10}: mean={result[f'{col}_mean']:.4f} | std={result[f'{col}_std']:.4f}")
    print()

    return result


# ============================================
# 6. FUNCTION: EVALUASI MODEL DI DATA TEST
# ============================================
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Fungsi untuk mengevaluasi model dan menampilkan metrik performa di data test,
    termasuk ROC-AUC. Perhitungan metrik memakai compute_metrics() yang sama
    dengan yang dipakai di cross_validate_model(), supaya konsisten.

    Parameters:
    - model: model yang sudah di-training
    - X_test, y_test: data test
    - model_name: nama model untuk label output

    Returns:
    - dict berisi metrik evaluasi
    """
    y_pred = model.predict(X_test)
    metrics = compute_metrics(model, X_test, y_test)

    print(f"===== Evaluasi {model_name} (Test Set) =====")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}" if not np.isnan(metrics['roc_auc']) else "ROC-AUC   : Tidak tersedia")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\n")

    return {
        "model_name": model_name,
        **metrics
    }


# ============================================
# 7. FUNCTION UTAMA: GABUNGKAN SEMUA PIPELINE
# ============================================
def run_all_models(df_train, df_test, target_col, cv=5):
    """
    Fungsi utama untuk menjalankan Logistic Regression, SVM, dan XGBoost
    menggunakan data train & test yang SUDAH terpisah.

    Alur:
    1. Fit encoding & scaling di data train, transform ke test
    2. Cross-validation 5-fold di data train -> hasil VALIDASI (metrik lengkap)
    3. Fit model final ke seluruh data train
    4. Evaluasi model ke data test -> hasil TEST (metrik lengkap)

    PENTING: hasil validasi dan hasil test TIDAK digabung jadi satu baris.
    Keduanya dikembalikan sebagai dua DataFrame terpisah.

    Parameters:
    - df_train: pandas DataFrame, dataset training
    - df_test: pandas DataFrame, dataset testing
    - target_col: str, nama kolom target
    - cv: int, jumlah fold cross-validation

    Returns:
    - dict: {"validation": DataFrame, "test": DataFrame}
    """
    # 1. Persiapan data (fit di train, transform ke test)
    X_train, X_test, y_train, y_test = prepare_data(df_train, df_test, target_col)

    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    validation_results = []
    test_results = []

    for name, model in models.items():
        print(f"===== {name} =====")

        # 2. Cross-validation di data training -> hasil VALIDASI
        print(f"--- Validasi ({cv}-fold Cross-Validation, dari data training) ---")
        cv_result = cross_validate_model(model, X_train, y_train, cv=cv)
        validation_results.append({
            "model_name": name,
            "accuracy": cv_result["accuracy_mean"],
            "precision": cv_result["precision_mean"],
            "recall": cv_result["recall_mean"],
            "f1_score": cv_result["f1_score_mean"],
            "roc_auc": cv_result["roc_auc_mean"],
            "accuracy_std": cv_result["accuracy_std"],
            "precision_std": cv_result["precision_std"],
            "recall_std": cv_result["recall_std"],
            "f1_score_std": cv_result["f1_score_std"],
            "roc_auc_std": cv_result["roc_auc_std"],
        })

        # 3. Fit model final ke seluruh data training
        model.fit(X_train, y_train)

        # 4. Evaluasi di data test -> hasil TEST
        eval_result = evaluate_model(model, X_test, y_test, name)
        test_results.append(eval_result)

    # 5. Ringkasan hasil, dipisah: validasi sendiri, test sendiri
    validation_df = pd.DataFrame(validation_results)
    test_df = pd.DataFrame(test_results)

    print("===== Ringkasan VALIDASI (Cross-Validation, dari data training) =====")
    print(validation_df)
    print()
    print("===== Ringkasan TEST SET =====")
    print(test_df)

    return {"validation": validation_df, "test": test_df}


# ============================================
# 8. FUNCTION: BACA TARGET COLUMN DARI FILE JSON (datasets/Info/*.json)
# ============================================
# Path RELATIVE terhadap folder utama (Test-Klasifikasi), tidak berubah
# meski nama folder utama diganti, selama script dijalankan dari dalamnya.
INFO_DIR = os.path.join("datasets", "Info")


def load_dataset_info(info_name, info_dir=INFO_DIR):
    """
    Membaca file JSON metadata dataset, misal datasets/Info/shoppers.json:
    {
        "name": "shoppers",
        "num_col_idx": [0,1,2,...],
        "cat_col_idx": [10,11,...],
        "target_col_idx": [17]
    }

    Parameters:
    - info_name: str, nama file json tanpa ekstensi (contoh: "shoppers")
    - info_dir: str, folder tempat file json disimpan

    Returns:
    - dict berisi isi json
    """
    path = os.path.join(info_dir, f"{info_name}.json")
    with open(path, "r") as f:
        info = json.load(f)
    return info


def get_target_col_name(df, info):
    """
    Menentukan nama kolom target berdasarkan index yang tertulis di JSON
    (info["target_col_idx"]), disesuaikan dengan urutan kolom aktual di df.

    Kalau posisi target berubah, cukup ubah angka "target_col_idx" di file
    JSON-nya, TIDAK perlu ubah kode ini.

    Parameters:
    - df: pandas DataFrame (biasanya df_train)
    - info: dict hasil load_dataset_info()

    Returns:
    - str, nama kolom target
    """
    target_idx = info["target_col_idx"][0]
    target_col = df.columns[target_idx]
    return target_col


# ============================================
# 9. GENERATE DATASET LIST OTOMATIS (RELATIVE PATH)
# ============================================
# Path RELATIVE terhadap folder utama (misal "Test-Klasifikasi").
# Kalau folder utama diganti nama, TIDAK perlu ubah kode ini,
# asal script tetap dijalankan dari dalam folder utama tsb.
# Sub-struktur folder (MCAR/60/CSV) diasumsikan sama untuk semua dataset;
# yang beda cuma nama folder dataset-nya sendiri (SHOPPERS, ADULT, dst).
N_FILES = 10  # jumlah file per kategori (index 0 - 9)


def generate_dataset_list(folder_name, info_name, n_files=N_FILES,
                           sub_path=("MCAR", "60", "CSV")):
    """
    Generate list dataset otomatis untuk kategori 'diff' dan 'mrmd',
    untuk SATU dataset (misal SHOPPERS atau ADULT).

    Pola nama file:
    - diff : train_impute_{i}.csv       / test_impute_{i}.csv
    - mrmd : train_impute_mrmd_{i}.csv  / test_impute_mrmd_{i}.csv

    Parameters:
    - folder_name: str, nama folder dataset di level utama, contoh "SHOPPERS", "ADULT"
    - info_name: str, nama file json (tanpa .json) di datasets/Info/, contoh "shoppers", "adult"
    - n_files: int, jumlah file per kategori (default 10, index 0-9)
    - sub_path: tuple, sub-folder di bawah folder_name sebelum ke CSV, default ("MCAR","60","CSV")

    Returns:
    - list of dict, siap digabung dengan hasil generate_dataset_list() dataset lain
    """
    base_dir = os.path.join(folder_name, *sub_path)
    dataset_list = []

    # ---- Kategori: diff ----
    for i in range(n_files):
        dataset_list.append({
            "name": f"{info_name}_diff_{i}",
            "train_path": os.path.join(base_dir, f"train_impute_{i}.csv"),
            "test_path": os.path.join(base_dir, f"test_impute_{i}.csv"),
            "info_name": info_name,
        })

    # ---- Kategori: mrmd ----
    for i in range(n_files):
        dataset_list.append({
            "name": f"{info_name}_mrmd_{i}",
            "train_path": os.path.join(base_dir, f"train_impute_mrmd_{i}.csv"),
            "test_path": os.path.join(base_dir, f"test_impute_mrmd_{i}.csv"),
            "info_name": info_name,
        })

    # Kategori 'mice' belum ada, jadi diabaikan dulu.
    # Kalau nanti sudah tersedia, tinggal tambahkan blok loop serupa di sini.

    return dataset_list


# ============================================
# 10. FUNCTION: LOOPING SEMUA DATASET + SIMPAN HASIL KE TXT
# ============================================
def run_multiple_datasets(dataset_list, cv=5, output_file="hasil_klasifikasi.txt"):
    """
    Menjalankan run_all_models() untuk beberapa dataset sekaligus (looping),
    lalu menyimpan seluruh log proses + ringkasan hasil ke satu file .txt.

    Hasil VALIDASI (cross-validation dari data training) dan hasil TEST SET
    disimpan & ditampilkan TERPISAH — tidak digabung jadi satu tabel.

    Kolom target TIDAK di-hardcode, melainkan ditentukan otomatis dari file
    JSON di datasets/Info/{info_name}.json (lihat load_dataset_info &
    get_target_col_name). Kalau posisi target berubah, cukup edit JSON-nya.

    Parameters:
    - dataset_list: list of dict, tiap dict berisi:
        {
            "name": "nama_dataset",
            "train_path": "path/ke/train.csv",
            "test_path": "path/ke/test.csv",
            "info_name": "nama_file_json_tanpa_ekstensi"   # contoh: "shoppers"
        }
    - cv: int, jumlah fold cross-validation
    - output_file: str, nama file txt untuk menyimpan hasil

    Returns:
    - dict, key = nama dataset, value = {"validation": DataFrame, "test": DataFrame}
    """
    all_summaries = {}

    with open(output_file, "w") as f:
        with redirect_stdout(f):
            print("=" * 60)
            print(f"LOG HASIL KLASIFIKASI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            print()

            for dataset in dataset_list:
                name = dataset["name"]
                train_path = dataset["train_path"]
                test_path = dataset["test_path"]
                info_name = dataset["info_name"]

                print("#" * 60)
                print(f"# DATASET: {name}")
                print("#" * 60)
                print()

                df_train = pd.read_csv(train_path)
                df_test = pd.read_csv(test_path)

                # ---- Tentukan kolom target dari JSON (datasets/Info/{info_name}.json) ----
                info = load_dataset_info(info_name)
                target_col = get_target_col_name(df_train, info)
                print(f"Target column (dari {info_name}.json, index {info['target_col_idx']}): '{target_col}'\n")

                result = run_all_models(df_train, df_test, target_col=target_col, cv=cv)
                all_summaries[name] = result

                print("\n\n")

            # ---- Ringkasan akhir: VALIDASI semua dataset digabung jadi satu tabel ----
            print("=" * 60)
            print("RINGKASAN AKHIR - VALIDASI (Cross-Validation, dari data training)")
            print("=" * 60)
            validation_rows = []
            for name, result in all_summaries.items():
                df_val = result["validation"].copy()
                df_val.insert(0, "dataset", name)
                validation_rows.append(df_val)
            validation_summary_df = pd.concat(validation_rows, ignore_index=True)
            print(validation_summary_df)
            print()

            # ---- Ringkasan akhir: TEST SET semua dataset digabung jadi satu tabel ----
            print("=" * 60)
            print("RINGKASAN AKHIR - TEST SET")
            print("=" * 60)
            test_rows = []
            for name, result in all_summaries.items():
                df_test_res = result["test"].copy()
                df_test_res.insert(0, "dataset", name)
                test_rows.append(df_test_res)
            test_summary_df = pd.concat(test_rows, ignore_index=True)
            print(test_summary_df)

    # Tampilkan juga di console bahwa proses selesai (di luar redirect, supaya kelihatan)
    print(f"Selesai. Hasil lengkap tersimpan di: {output_file}")

    return all_summaries


# ============================================
# 11. PEMANGGILAN: LOOPING KE SEMUA DATASET
# ============================================
if __name__ == "__main__":

    # Gabungkan dataset SHOPPERS dan ADULT (keduanya pakai pola path/file yang sama).
    # Kalau mau tambah dataset lain lagi, tinggal tambahkan pemanggilan
    # generate_dataset_list() baru dan gabungkan pakai "+".
    dataset_list = (
        generate_dataset_list(folder_name="SHOPPERS", info_name="shoppers")
        # + generate_dataset_list(folder_name="ADULT", info_name="adult")
    )

    all_summaries = run_multiple_datasets(
        dataset_list,
        cv=5,
        output_file="hasil_klasifikasi.txt"
    )