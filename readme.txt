The main script uses the webcam to detect 4 emotions:
- neutral
- happy
- fear
- angry

For facial detection, haarcascade-frontalface is used to extract bounding box coordinates

The detected faces are then cropped and fed into the model

The prediction with the highest probability is used 

