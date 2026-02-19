from core.search import search_image

def predict(image_path):
    label, distance = search_image(image_path)

    if label == "NEW_CLASS":
        print(f"🚀 New class detected! Distance: {distance}")
    else:
        print(f"✅ Belongs to: {label} | Distance: {distance}")
