from typing import List
from pydantic import BaseModel

class ImageSample(BaseModel):
    pixels: List[List[List[float]]]
    label: int

class ImagesBatchResponse(BaseModel):
    images: List[ImageSample]
    eof: bool
    
class Predictions(BaseModel):
    probabilities: List[float]
    batch_index: int
    eof: bool
