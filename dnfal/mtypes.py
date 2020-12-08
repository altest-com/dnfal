from base64 import b64encode
from typing import List, Tuple

import cv2 as cv
import numpy as np


class Frame:
    def __init__(
        self,
        image: np.ndarray,
        key: int = None
    ):
        self.key: int = key
        self.data: dict = {}
        self.image: np.ndarray = image

    def serialize(self):
        return {
            'key': self.key,
            'image': self.image_bytes
        }

    @property
    def image_bytes(self):
        return cv.imencode('.jpg', self.image)[1]


class _Entity:
    def __init__(
        self,
        box: tuple,
        image: np.ndarray,
        key: int = None,
        frame: Frame = None,
        embeddings: np.ndarray = None,
        detect_score: float = 0.0,
        timestamp: float = 0,
        offset: Tuple[int, int] = (0, 0)
    ):
        self.key: int = key
        self.data: dict = {}
        self.box: tuple = box
        self.image: np.ndarray = image
        self.frame: Frame = frame
        self.embeddings: np.ndarray = embeddings
        self.detect_score: float = detect_score
        self.timestamp: float = timestamp
        self.offset: Tuple[int, int] = offset

    def serialize(self):

        frame = None
        if self.frame is not None:
            frame = self.frame.serialize()

        return {
            'box': self.box,
            'image': self.image_bytes(),
            'frame': frame,
            'embeddings': self.embeddings.tolist(),
            'timestamp': self.timestamp
        }

    def image_bytes(self, fmt: str = '.jpg'):
        return cv.imencode(fmt, self.image)[1]

    def image_base64(self, fmt: str = '.jpg') -> str:
        return str(b64encode(self.image_bytes(fmt),), 'utf-8')

    def __str__(self):
        return f'entity ({100 * self.detect_score:.1f}%)'

    def __repr__(self):
        return self.__str__()


class Face(_Entity):
    def __init__(
        self,
        box: tuple,
        image: np.ndarray,
        aligned_image: np.ndarray = None,
        key: int = None,
        frame: Frame = None,
        subject: 'Subject' = None,
        landmarks: np.ndarray = None,
        embeddings: np.ndarray = None,
        detect_score: float = 0.0,
        mark_score: float = 0.0,
        nose_deviation: Tuple[float, float] = (0.0, 0.0),
        timestamp: float = 0,
        offset: Tuple[int, int] = (0, 0)
    ):
        super().__init__(
            box=box,
            image=image,
            key=key,
            frame=frame,
            embeddings=embeddings,
            detect_score=detect_score,
            timestamp=timestamp,
            offset=offset
        )

        self.aligned_image: np.ndarray = aligned_image
        self.subject: 'Subject' = subject
        self.landmarks: np.ndarray = landmarks
        self.mark_score: float = mark_score
        self.nose_deviation: Tuple[float, float] = nose_deviation

    def serialize(self):
        return {**super().serialize(), 'landmarks': self.landmarks.tolist()}

    def __str__(self):
        return f'face ({100 * self.detect_score:.1f}%)'


class Body(_Entity):
    def __init__(
        self,
        box: tuple,
        image: np.ndarray,
        key: int = None,
        frame: Frame = None,
        subject: 'Subject' = None,
        embeddings: np.ndarray = None,
        detect_score: float = 0.0,
        timestamp: float = 0,
        offset: Tuple[int, int] = (0, 0)
    ):
        super().__init__(
            box=box,
            image=image,
            key=key,
            frame=frame,
            embeddings=embeddings,
            detect_score=detect_score,
            timestamp=timestamp,
            offset=offset
        )

        self.subject: 'Subject' = subject

    def __str__(self):
        return f'body ({100 * self.detect_score:.1f}%)'

    def __repr__(self):
        return self.__str__()


class Subject:
    def __init__(
        self,
        faces: List[Face],
        embeddings: np.ndarray,
        key: int = None
    ):
        self.key: int = key
        self.data: dict = {}
        self.faces: List[Face] = faces
        self.embeddings: np.ndarray = embeddings
        self.last_updated: float = -float('inf')

    def append_face(self, face: Face):
        self.faces.append(face)
        if face.timestamp > self.last_updated:
            self.last_updated = face.timestamp

    def serialize(self):
        return {
            'faces': [face.serialize() for face in self.faces],
            'embeddings': self.embeddings.tolist()
        }
