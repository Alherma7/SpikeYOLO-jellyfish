import snntorch
from snntorch import spikegen

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
import numpy as np
from torchvision import transforms

class CustomDataset(Dataset):
    """Dataset personalizado para redes neuronales de impulsos (SNN) y YOLO.

    Se encarga de cargar las imágenes y etiquetas en formato YOLO, aplicar la
    función Canny y codificar las imágenes a impulsos con la librería snntorch.

    Attributes:

        img_dir (str): directorio origen con las imágenes

        label_dir (str): direcotrio origen con las anotaciones

        timestep (int): número de pasos temporales para la codificación a impulsos

        edge (bool): Indica si aplicar el algoritmo Canny o no.

        imgsz (int): resolución (cuadrada) a la que se redimensionan las imágenes.
            Debe coincidir con la resolución usada durante entrenamiento/validación.

        to_tensor (transforms.ToTensor): Transforma las imágenes a tensores.

        resize (transforms.resize): Transforma las imágenes a la resolución `imgsz`.

        img_files (list): Lista con las imágenes válidas.

        rate_encode (str): Indica si usar codificación rate_encode.

        rate_conv (str): Indica si usar codificación rate_conv.

        rate_kwargs (dict): Argumentos adicionales para las funciones de codificación.
    """

    def __init__(self, img_dir, label_dir, timestep=4, edge=True,
                 rate_encode=True, rate_conv=False,
                 rate_kwargs=None, imgsz=320):
        """Inicializa las rutas, transformaciones y configuraciones

        Args:

            img_dir (str): directorio origen con las imágenes

            label_dir (str): direcotrio origen con las anotaciones

            timestep (int, optional): número de pasos temporales para la codificación a impulsos.
                Por defecto es 4.

            edge (bool, optional): Indica si aplicar el algoritmo Canny. Por defecto es True.
                Nota: `edge=True` produce imágenes de 1 canal (escala de grises), lo que debe
                coincidir con `ch: 1` en el yaml del modelo. Si se cambia a `edge=False` (RGB
                crudo), el yaml del modelo debe pasar a `ch: 3`, o la primera capa `MS_DownSampling`
                fallará por desajuste de canales.

            rate_encode (bool, optional): Indica si aplicar rate_encode. Por defecto es True.

            rate_conv (bool, optional): Indica si aplicar rate_encode. Por defecto es False.

            rate_kwargs (dict, optional): Argumentos adicionales para las funciones de codificación.
                Por defecto es None.

            imgsz (int, optional): resolución cuadrada de redimensionado. Por defecto 320.
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.timestep = timestep
        self.edge = edge
        self.imgsz = imgsz
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((imgsz, imgsz))
        self.img_files = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))
        self.rate_encode = rate_encode
        self.rate_kwargs = rate_kwargs if rate_kwargs is not None else {}
        self.rate_conv = rate_conv


    def __len__(self):
        """Devuelve el número total de imágenes del dataset.
        """
        return len(self.img_files)

    def __getitem__(self, idx):
        """Carga, procesa y codifica a impulsos el dataset según su índice.

        Realiza la lectura de la ruta, aplica la detección de bordes, redimensiona la imagen y
        transforma a un tensor de impulso asociandolo a sus anotaciones.

        Args:

            idx(int): Índice del elemento a escoger.

        Returns:

            dict: un diccionario por muestra con las claves que espera el resto del pipeline
            de ultralytics (`v8DetectionLoss`, `DetectionValidator`):

            - img: Tensor con las dimensiones [T,C,H,W] (impulsos).

            - cls: Tensor [N,1] con la clase de cada caja.

            - bboxes: Tensor [N,4] con las cajas normalizadas (cx,cy,w,h) en [0,1].

            - ori_shape: (H0,W0) resolución original de la imagen.

            - resized_shape: (H,W) resolución tras el resize (=`imgsz`).

            - im_file: ruta de la imagen original.
        """

        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        image = cv2.imread(img_path)
        h0, w0 = image.shape[:2]  # orig hw

        if self.edge:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = canny(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        image = self.resize(image)
        image = self.to_tensor(image)  # [C,H,W] in [0,1]

        if self.rate_encode:
            image = spikegen.rate(image, num_steps=self.timestep, **self.rate_kwargs)  # [T,C,H,W]
        elif self.rate_conv:
            image = spikegen.rate_conv(image)
            image = image.unsqueeze(0).repeat(self.timestep, 1, 1, 1)  # [T,C,H,W]

        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    bbox = list(map(float, parts[1:5]))
                    labels.append([cls] + bbox)

        if labels:
            labels = torch.tensor(labels, dtype=torch.float32)
            cls = labels[:, 0:1]
            bboxes = labels[:, 1:5]
        else:
            cls = torch.zeros((0, 1), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)

        return {
            'img': image,
            'cls': cls,
            'bboxes': bboxes,
            'ori_shape': (h0, w0),
            'resized_shape': (self.imgsz, self.imgsz),
            'im_file': img_path,
        }

def collate_fn(batch):
    """Función personalizada para agrupar los batches del Dataloader.

    Apila las imágenes en un único tensor `[B,T,C,H,W]` (batch-first: el modelo se
    encarga internamente de traponer a `[T,B,C,H,W]` cuando `MS_GetT` está deshabilitado,
    ver `BaseModel._predict_once` en `ultralytics/nn/tasks.py`) y concatena las cajas de
    TODAS las imágenes (no solo la primera) añadiendo el índice de imagen dentro del batch,
    tal y como espera `v8DetectionLoss`/`DetectionValidator`.

    La evaluación (mAP) se realiza en el espacio de la imagen redimensionada (`imgsz`x`imgsz`):
    `ori_shape` se fija igual a `resized_shape` y se pasa un `ratio_pad` identidad, evitando
    tener que modelar el resize (no letterbox, sin padding) que aplica este dataset dentro de
    `ops.scale_boxes`.

    Args:

        batch (list): Lista de diccionarios provenientes del Dataloader (ver `CustomDataset.__getitem__`).

    Returns:

        dict: batch con las claves `img`, `cls`, `bboxes`, `batch_idx`, `ori_shape`,
        `resized_shape`, `im_file`, `ratio_pad`.
    """

    imgs = torch.stack([b['img'] for b in batch], 0)  # [B,T,C,H,W]

    cls_list, bbox_list, batch_idx_list = [], [], []
    for i, b in enumerate(batch):
        n = b['cls'].shape[0]
        cls_list.append(b['cls'])
        bbox_list.append(b['bboxes'])
        batch_idx_list.append(torch.full((n,), i, dtype=torch.float32))

    cls = torch.cat(cls_list, 0) if cls_list else torch.zeros((0, 1), dtype=torch.float32)
    bboxes = torch.cat(bbox_list, 0) if bbox_list else torch.zeros((0, 4), dtype=torch.float32)
    batch_idx = torch.cat(batch_idx_list, 0) if batch_idx_list else torch.zeros((0,), dtype=torch.float32)

    return {
        'img': imgs,
        'cls': cls,
        'bboxes': bboxes,
        'batch_idx': batch_idx,
        'ori_shape': [b['resized_shape'] for b in batch],  # eval in resized-image space, see docstring
        'resized_shape': [b['resized_shape'] for b in batch],
        'im_file': [b['im_file'] for b in batch],
        'ratio_pad': [((1.0, 1.0), (0.0, 0.0)) for _ in batch],  # identity: plain resize, no letterbox pad
    }

def canny(img_gray):
    """Algoritmo optimizadao de detección de bordes mediante Canny.

    Aplica normalización MinMax, mejora el contraste mediante ecualización adaptativa
    (CLAHE), reduce el ruido con un filtro Gaussiano y calcula los umbrales de histéresis basándose
    en la mediana de la imagen. Finalmente, dilata los bordes.

    Args:

        img_gray(np.ndarray): Imagen de entrada en escala de grises.

    Returns:

        np.ndarray: Imagen binaria con el filtro de bordes.
    """

    # Normalizar histograma para estandarizar antes del CLAHE
    img_normalized = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_normalized)
    blurred = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
    median = np.median(blurred)
    lower = int(max(0, 0.2 * median))
    upper = int(min(255, 0.6 * median))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = np.ones((1,1), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)
