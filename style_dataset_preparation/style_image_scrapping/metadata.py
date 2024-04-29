from PIL import Image
from PIL.ExifTags import TAGS

imagename = "photos/art nouveau/artnouveau25.jpeg"

image = Image.open(imagename)

info_dict = {
    "Filename": image.filename,
    "image Size": image.size,
    "image Height": image.height,
    "image Width": image.width,
    "image Format": image.format,
    "image Mode": image.mode,
    "image is Animated": getattr(image, "is_animated", False),
    "Frames in image": getattr(image, "n_frames", 1)
}

for label,value in info_dict.items():
    print(f"{label:25}: {value}")

exifdata = image.getexif()
for tag_id in exifdata:
    tag = TAGS.get(tag_id, tag_id)
    data = exifdata.get(tag_id)
    if isinstance(data, bytes):
        data = data.decode()
    print(f"{tag:25}: {data}")
