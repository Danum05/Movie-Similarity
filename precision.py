import json

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def label_relevancy(rekomendasi, groundtruth):
    gt_titles = {item['title'].lower() for item in groundtruth}
    for item in rekomendasi:
        if item['title'].lower() in gt_titles:
            item['relevan'] = "1"
        else:
            item['relevan'] = "0"
    return rekomendasi

def calculate_metrics(labeled_results, groundtruth):
    tp = sum(1 for result in labeled_results if result['relevan'] == "1")
    fp = sum(1 for result in labeled_results if result['relevan'] == "0")

    fn = len(groundtruth) - tp  # Jumlah groundtruth - yang ditemukan benar
    tn = 0  # Tidak bisa dihitung karena tidak ada info relevansi di groundtruth

    precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score, tn

# === MAIN ===
knn_results_path = 'Rekomendasi.json'
groundtruth_path = 'groundtruth.json'

# Load data
rekomendasi = load_data(knn_results_path)
groundtruth = load_data(groundtruth_path)

# Tandai relevansi berdasarkan title
labeled_rekomendasi = label_relevancy(rekomendasi, groundtruth)

# (Opsional) simpan hasil yang sudah ditandai
save_data(knn_results_path, labeled_rekomendasi)

# Hitung metrik
precision, recall, f1_score, tn = calculate_metrics(labeled_rekomendasi, groundtruth)

# Tampilkan hasil
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1_score:.2f}%")
