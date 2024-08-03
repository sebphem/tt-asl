import torchvision.transforms as transforms
from PIL import Image

# Define the custom transform
class CenterCropAndZoom:
    def __init__(self, crop_size, zoom_factor):
        self.crop_size = crop_size
        self.zoom_factor = zoom_factor

    def __call__(self, img):
        # Center crop
        crop_transform = transforms.CenterCrop(self.crop_size)
        img = crop_transform(img)

        # Zoom in
        width, height = img.size
        new_size = (int(width * self.zoom_factor), int(height * self.zoom_factor))
        resize_transform = transforms.Resize(new_size)
        img = resize_transform(img)

        return img

def crop_and_zoom(image: str):
    
    image = Image.open(image)
    # Define the crop size (smallest dimension to ensure 1:1 aspect ratio) and zoom factor (60%)
    crop_size = min(image.size)
    zoom_factor = 0.6

    # Create the transform and apply it
    transform = CenterCropAndZoom(crop_size, zoom_factor)
    transformed_image = transform(image)
    
    return transformed_image