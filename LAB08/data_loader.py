import os
from pathlib import Path
import cv2
import numpy as np

CLASS_NAMES = ["abyssinian", "cyprus", "lykoi", "donskoy", "chausie"]
IMAGE_SIZE = (100, 100)
MAX_PER_CLASS = 200

def find_dataset_dir():
    """Find the dataset folder containing one subfolder per cat breed."""
    candidates = []
    env_path = os.environ.get("CAT_BREEDS_DATA")
    if env_path:
        candidates.append(Path(env_path))

    project_dir = Path(__file__).resolve().parent
    candidates.extend([
        project_dir / "data-animal" / "cat-breeds" / "cat-breeds",
        project_dir / "data-animal" / "cat-breeds",
        Path(r"D:\ML\LAB05\data-animal\cat-breeds\cat-breeds"),
        Path(r"D:\ML\LAB05\data-animal\cat-breeds"),
    ])

    for candidate in candidates:
        if candidate.exists() and all((candidate / name).is_dir() for name in CLASS_NAMES):
            return candidate

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "ไม่พบโฟลเดอร์ Dataset ที่มีโฟลเดอร์สายพันธุ์แมวครบ 5 คลาส\n"
        "ตรวจสอบตำแหน่ง Dataset หรือกำหนดตัวแปร CAT_BREEDS_DATA\n"
        f"ตำแหน่งที่ตรวจแล้ว:\n{checked}"
    )

def load_images(dataset_dir=None, image_size=IMAGE_SIZE, max_per_class=MAX_PER_CLASS):
    """Load images, resize to 100x100, convert BGR to RGB, and assign folder labels."""
    dataset_dir = Path(dataset_dir) if dataset_dir else find_dataset_dir()
    images, labels, paths = [], [], []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"ไม่พบโฟลเดอร์คลาส: {class_dir}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )[:max_per_class]

        if not image_files:
            raise ValueError(f"ไม่พบไฟล์รูปภาพในโฟลเดอร์: {class_dir}")

        loaded_count = 0
        for image_path in image_files:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"ข้ามไฟล์ที่อ่านไม่ได้: {image_path}")
                continue
            image = cv2.resize(image, image_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            labels.append(label)
            paths.append(str(image_path))
            loaded_count += 1

        print(f"{class_name}: โหลดได้ {loaded_count} ภาพ")

    if not images:
        raise ValueError("ไม่สามารถโหลดรูปภาพได้")
    return np.asarray(images, dtype=np.uint8), np.asarray(labels, dtype=np.int64), paths
