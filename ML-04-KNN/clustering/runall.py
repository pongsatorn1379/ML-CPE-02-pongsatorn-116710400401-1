import subprocess

files = ["data_loader.py", "kmeans_tf.py", "knn_tools.py","main.py","visualize.py"]

for f in files:
    print(f"กำลังรัน: {f}")
    subprocess.run(["python", f])