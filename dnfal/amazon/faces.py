from typing import List

import boto3
import numpy as np
import cv2 as cv

MIN_IMAGE_WIDTH = 80
MAX_IMAGE_FACES = 100
MIN_FACE_WIDTH = 64
MAX_FACE_WIDTH = 164


class AwsFace:

    GENDER_MAN = 'male'
    GENDER_WOMAN = 'woman'

    def __init__(self):
        self.box: tuple = ()
        self.age_low: int = 0
        self.age_high: int = 0
        self.has_beard: bool = False
        self.beard_score: float = 0
        self.gender: str = ''
        self.gender_score: float = 0
        self.has_mustache: bool = False
        self.mustache_score: float = 0


_GENDER_CHOICES = {
    'Male': AwsFace.GENDER_MAN,
    'Female': AwsFace.GENDER_WOMAN
}


def _resize_image(image, min_width: int = None, max_width: int = None):
    height, width = image.shape[0:2]
    if max_width is not None and width > max_width:
        scale = max_width / width
        image = cv.resize(image, (max_width, int(scale * height)))
    elif min_width is not None and width < min_width:
        scale = min_width / width
        image = cv.resize(image, (min_width, int(scale * height)))

    return image


def _build_mosaic(images: List[np.ndarray]):

    n_images = len(images)
    images_ = []
    max_width = 0
    max_height = 0

    for image in images:
        image = _resize_image(
            image,
            min_width=MIN_FACE_WIDTH,
            max_width=MAX_FACE_WIDTH
        )
        images_.append(image)

        height, width = image.shape[0:2]
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width

    images = images_

    n_rows = n_cols = int(n_images ** 0.5 + 1)
    if n_rows * (n_cols - 1) >= n_images:
        n_cols = n_cols - 1

    canvas_shape = (n_rows * max_height, n_cols * max_width, 3)
    image_mosaic = np.zeros(canvas_shape, dtype=np.uint8)

    image_ind = 0
    for row in range(n_rows):
        i = row * max_height
        for col in range(n_cols):
            j = col * max_width
            image = images[image_ind]
            height, width = image.shape[0:2]
            image_mosaic[i:(i + height), j:(j + width), :] = image[:]
            image_ind += 1

            if image_ind > n_images - 1:
                break

        if image_ind > n_images - 1:
            break

    return image_mosaic, n_rows, n_cols


def analyze_faces(images: List[np.ndarray]) -> List[AwsFace]:

    if not len(images):
        return []

    n_images = len(images)
    if n_images > 1:
        image_mosaic, n_rows, n_cols = _build_mosaic(images)
        image_mosaic = _resize_image(image_mosaic, min_width=MIN_IMAGE_WIDTH)
    else:
        image_mosaic = _resize_image(
            images[0],
            min_width=MIN_IMAGE_WIDTH,
            max_width=MAX_FACE_WIDTH
        )
        n_rows = n_cols = 1

    client = boto3.client('rekognition')

    _, image_bytes = cv.imencode('.jpg', image_mosaic)
    image_bytes = image_bytes.tobytes()

    response = client.detect_faces(
        Image={'Bytes': image_bytes},
        Attributes=['ALL']
    )

    faces = [None] * n_images

    mosaic_height, mosaic_width = image_mosaic.shape[:2]
    cell_width = mosaic_width / n_cols
    cell_height = mosaic_height / n_rows

    for df in response['FaceDetails']:
        box_top = float(df['BoundingBox']['Top'])
        box_left = float(df['BoundingBox']['Left'])
        box_width = float(df['BoundingBox']['Width'])
        box_height = float(df['BoundingBox']['Height'])

        row = int(mosaic_height * box_top / cell_height)
        col = int(mosaic_width * box_left / cell_width)
        image_ind = row * n_cols + col

        # cv.putText(
        #     image_mosaic,
        #     f'{row}, {col}, {image_ind}',
        #     (int(mosaic_width * box_left), int(mosaic_height * box_top)),
        #     color=(255,),
        #     fontFace=cv.FONT_HERSHEY_PLAIN,
        #     fontScale=1
        # )
        #
        # cv.rectangle(
        #     image_mosaic,
        #     pt1=(int(mosaic_width * box_left), int(mosaic_height * box_top)),
        #     pt2=(int(mosaic_width * (box_left + box_width)), int(mosaic_height * (box_top + box_height))),
        #     color=(255,),
        # )

        if image_ind < n_images:
            store = True
            if faces[image_ind] is not None:
                box = faces[image_ind].box
                area = (box[2] - box[0]) * (box[3] - box[1])
                if box_width * box_height < area:
                    store = False

            if store:
                face = AwsFace()
                face.box = (
                    box_left,
                    box_top,
                    box_left + box_width,
                    box_top + box_height,
                )
                face.age_low = int(df['AgeRange']['Low'])
                face.age_high = int(df['AgeRange']['High'])
                face.gender = _GENDER_CHOICES[str(df['Gender']['Value'])]
                face.gender_score = float(df['Gender']['Confidence'])/100
                face.has_beard = bool(df['Beard']['Value'])
                face.beard_score = float(df['Beard']['Confidence'])/100
                face.has_mustache = bool(df['Mustache']['Value'])
                face.mustache_score = float(df['Mustache']['Confidence'])/100
                # noinspection PyTypeChecker
                faces[image_ind] = face

            # cv.imshow('face', images[image_ind])
            # cv.waitKey()

    # cv.imshow('image', image_mosaic)
    # cv.waitKey()

    # noinspection PyTypeChecker
    return faces
