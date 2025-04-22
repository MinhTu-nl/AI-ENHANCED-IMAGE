import kagglehub

# Download latest version
path = kagglehub.dataset_download("soumikrakshit/dark-face-dataset")

print("Path to dataset files:", path)