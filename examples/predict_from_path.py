from face_id.api import FaceRecognizer

IMAGE_PATH = "/path/to/input.jpg"
GALLERY_PATH = "data/gallery.npz"

if __name__ == "__main__":
    recognizer = FaceRecognizer(device="gpu", model="buffalo_s", det_size=320, max_side=1280)
    results = recognizer.predict(
        input_path=IMAGE_PATH,
        gallery_path=GALLERY_PATH,
        threshold=0.38,
    )
    for r in results:
        print(f"Image:   {r.image}")
        print(f"Person:  {r.person}")
        print(f"Match:   {r.similarity_percent:.2f}%")
        print(f"Accepted: {r.accepted}")
