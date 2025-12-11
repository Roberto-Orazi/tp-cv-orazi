from PIL import Image
from pathlib import Path

input_folder = input("Ingresa la ruta de la carpeta con imagenes: ").strip()
input_folder = Path(input_folder)

output_folder = input_folder / "convertidas"
output_folder.mkdir(exist_ok=True)

image_files = list(input_folder.glob("*.*"))
image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"]

converted = 0
for img_file in image_files:
    if img_file.suffix.lower() in image_extensions:
        try:
            img = Image.open(img_file).convert("RGB")
            output_path = output_folder / f"{img_file.stem}.jpg"
            img.save(output_path, "JPEG", quality=95)
            print(f"Convertido: {img_file.name} -> {output_path.name}")
            converted += 1
        except Exception as e:
            print(f"Error con {img_file.name}: {e}")

print(f"\n{converted} imagenes convertidas a JPG en: {output_folder}")
