# Hand Gesture Recognition

Detect hand gestures using a webcam. Gestures are recognized in your browser
with MediaPipe; camera video is processed locally and is not uploaded.

## Run it

Camera access and JavaScript modules require a secure origin. From this folder, start the local web server:

```bash
python app.py
```

Then visit <http://localhost:8000> and choose **Start camera**. Allow camera
access when prompted.
The first start downloads the MediaPipe runtime and recognition model, so an
internet connection is needed.

## Gestures

The guide includes 30 single-hand gestures: fist, pinky, ring, ring + pinky,
middle + ring, middle + ring + pinky, one, index + pinky, index + ring + pinky,
two, two + pinky, three, four, like, call me, thumb + ring, thumb + ring +
pinky, thumb + middle + pinky, thumb + three fingers, L sign, love, thumb +
index + ring, thumb + index + ring + pinky, thumb + index + middle, thumb +
index + middle + pinky, thumb + four fingers, open palm, nope, pinch, and OK.

Gesture poses are recognized from hand landmarks in the browser. The 27
finger-extension poses use hand-relative landmark distances; pinch, OK, and
thumbs-down have dedicated detection. Some finger combinations are more
sensitive to hand angle and visibility than familiar signs. For best results,
face the camera in a well-lit space and hold one hand clearly in frame. The
preview is mirrored to feel like a mirror; recognition works on camera frames
directly.
