from grepl.inference import get_mmpose_vitpose_service
svc = get_mmpose_vitpose_service()

path = "/home/rocus/Documents/john/grepl/clip_frames/-Hc7Mkb8SEI_352_30/frame_000000000011.jpg" # ruotolo vs langaker

prediction = svc.predict(path)
print(prediction)