import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import json
from pathlib import Path

from utils.dataset import DogDataset

csv_file = "../dataset/dogs.csv"
temp_dataset = DogDataset(csv_file, "../dataset", "train")
all_breeds = sorted(temp_dataset.label_to_idx.keys())

print("Herramienta de Anotacion Manual - INSTRUCCIONES:")
print("=" * 80)
print("1. Hacé DOS clicks en la imagen:")
print("   - 1er click: Esquina superior izquierda")
print("   - 2do click: Esquina inferior derecha")
print("   -> (Un recuadro verde aparecerá). Si te equivocás, NO se puede deshacer esa caja.")
print("      Tendrás que terminar con esa imagen y volver a empezar si es grave.")
print("2. Presioná la tecla 'n' para CONFIRMAR la caja.")
print("3. Recién ahí, MIRÁ ESTA CONSOLA: Te pedirá el número de la raza.")
print("4. Escribí el ID (ej: 15) y dale ENTER.")
print("5. Tecla 's': Guardar imagen completa y pasar a la siguiente.")
print("6. Tecla 'q': Salir.")
print("=" * 80)

input_folder = input("\nIngresa la ruta de la carpeta con las 10 imagenes: ").strip()
input_folder = Path(input_folder)

if not input_folder.exists():
    print(f"ERROR: La carpeta {input_folder} no existe")
    sys.exit(1)

image_files = (
    list(input_folder.glob("*.jpg"))
    + list(input_folder.glob("*.jpeg"))
    + list(input_folder.glob("*.png"))
)

if len(image_files) == 0:
    print("ERROR: No se encontraron imagenes")
    sys.exit(1)

print(f"\nEncontradas {len(image_files)} imagenes")

annotations_data = {"images": []}

current_point = None
current_box = []


def mouse_callback(event, x, y, flags, param):
    global current_point, current_box

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_box) == 0:
            current_box = [x, y]
            print(f"Esquina superior izquierda: ({x}, {y})")
        elif len(current_box) == 2:
            current_box.extend([x, y])
            print(f"Esquina inferior derecha: ({x}, {y})")
            print(f"Bounding box: {current_box}")


for img_idx, img_path in enumerate(image_files):
    print(f"\n{'='*80}")
    print(f"Imagen {img_idx+1}/{len(image_files)}: {img_path.name}")
    print("=" * 80)

    img = cv2.imread(str(img_path))
    img_copy = img.copy()

    cv2.namedWindow("Imagen")
    cv2.setMouseCallback("Imagen", mouse_callback)

    image_annotations = []

    while True:
        display_img = img_copy.copy()

        if len(current_box) == 2:
            cv2.circle(display_img, tuple(current_box), 5, (0, 255, 0), -1)
        elif len(current_box) == 4:
            cv2.rectangle(
                display_img,
                (current_box[0], current_box[1]),
                (current_box[2], current_box[3]),
                (0, 255, 0),
                2,
            )

        for ann in image_annotations:
            bbox = ann["bbox"]
            cv2.rectangle(
                display_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2
            )
            cv2.putText(
                display_img,
                ann["breed"],
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        cv2.imshow("Imagen", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nTerminando anotacion...")
            break
        elif key == ord("s"):
            print(f"\nImagen completada con {len(image_annotations)} anotaciones")
            break
        elif key == ord("n") and len(current_box) == 4:
            x1 = min(current_box[0], current_box[2])
            y1 = min(current_box[1], current_box[3])
            x2 = max(current_box[0], current_box[2])
            y2 = max(current_box[1], current_box[3])

            normalized_box = [x1, y1, x2, y2]

            print("\nRazas disponibles:")
            for i, breed in enumerate(all_breeds):
               print(f"{i}: {breed}")
            print("(Mirá la tabla de referencia que te pasé)")

            # Hack para Mac: cv2.waitKey(1) necesita llamarse para procesar eventos de ventana
            cv2.imshow("Imagen", display_img)
            cv2.waitKey(1)

            try:
                breed_idx = int(input("\nIngresa el numero de la raza (y dale Enter): "))
                if 0 <= breed_idx < len(all_breeds):
                    breed_name = all_breeds[breed_idx]
                else:
                    print("Numero invalido, usando 'Unknown'")
                    breed_name = "Unknown"
            except ValueError:
                 print("Entrada invalida, usando 'Unknown'")
                 breed_name = "Unknown"

            image_annotations.append({"bbox": normalized_box, "breed": breed_name})

            print(f"Anotacion guardada: {breed_name}")
            print("Presiona 'n' para siguiente perro o 's' para siguiente imagen")

            cv2.rectangle(
                img_copy,
                (current_box[0], current_box[1]),
                (current_box[2], current_box[3]),
                (255, 0, 0),
                2,
            )

            current_box = []

    if key == ord("q"):
        break

    annotations_data["images"].append(
        {"file": str(img_path), "annotations": image_annotations}
    )

    cv2.destroyAllWindows()

output_file = "anotaciones_manuales.json"
with open(output_file, "w") as f:
    json.dump(annotations_data, f, indent=2)

print(f"\n{'='*80}")
print("ANOTACION COMPLETADA")
print("=" * 80)
print(f"Total imagenes anotadas: {len(annotations_data['images'])}")
print(
    f"Total anotaciones: {sum(len(img['annotations']) for img in annotations_data['images'])}"
)
print(f"Guardado en: {output_file}")
