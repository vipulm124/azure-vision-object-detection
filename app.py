import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from PIL import Image
import os
import json
from dotenv import load_dotenv

# read values from .env file
load_dotenv()
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")


client = ComputerVisionClient(AZURE_VISION_ENDPOINT, CognitiveServicesCredentials(AZURE_VISION_KEY))

visual_features = [VisualFeatureTypes.objects, VisualFeatureTypes.color]

def resize_image(image_path, max_size=4 * 1024 * 1024, target_size=(2048, 2048)):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        if len(image_data) > max_size:
            image = Image.open(image_path)
            image.thumbnail(target_size, Image.LANCZOS)
            image.save(image_path, format=image.format)
            print(f"Resized image {image_path} to fit within the size limit.")
    return image_path


def detect_object(image_path, object_list):
    print(f"Detecting objects in {image_path}")
    
    image_path = resize_image(image_path)
    
    with open(image_path, "rb") as image_stream:
        result = client.analyze_image_in_stream(image_stream, visual_features)
        response = json.dumps(result.as_dict(), indent=4)
        write_to_file("=====================================\n")
        write_to_file(f"Image: {image_path}\n")
        write_to_file(f"ImageData: {response}\n")
        objects = result.objects
        for obj in objects:
            objects_split = obj.object_property.split(" ")
            if any(x in objects_split for x in object_list):
                print(f"Found {obj.object_property} in the image with confidence {obj.confidence}")


objects_list = ["keyboard", "computer", "laptop", "key", "cow", "mammal", "chair", "nature", "mountain", "ice", "hills"]

file_dir = "/Users/vipulmalhotra/Documents/source/repo/azure-vision-object-detection/images"

output_file = "output.txt"

def write_to_file(data):
    # append data to the file
    with open(output_file, "a") as file:
        # file.write("\n")
        file.write(data)


def loop_through_images(file_dir_path):
    print('Inside directory:', file_dir_path)
    for image in os.listdir(file_dir_path):
        time.sleep(6)
        print(image)
        if image == ".DS_Store":
            continue
        image_path = os.path.join(file_dir_path, image)
        print(f"Image with full path: {image_path}")
        if os.path.isdir(image_path):
            print(image_path)
            loop_through_images(image_path)
        else:
            detect_object(image_path, object_list=objects_list)


def empty_output_file():
    with open(output_file, "w") as file:
        file.write("")
        file.close()

empty_output_file()
loop_through_images(file_dir_path=file_dir)