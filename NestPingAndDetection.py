import firebase_admin
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from firebase_admin import messaging
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Google Nest API setup
NEST_API_URL = "https://smartdevicemanagement.googleapis.com/v1"
ACCESS_TOKEN = 'YOUR_NEST_ACCESS_TOKEN'  # Replace with your access token

# Firebase setup
FIREBASE_CREDENTIALS_PATH = 'path/to/your-firebase-adminsdk.json'  # Replace with your Firebase credentials path
firebase_admin.initialize_app(firebase_admin.credentials.Certificate(FIREBASE_CREDENTIALS_PATH))

# Load pre-trained Faster R-CNN model for person detection
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def get_nest_events():
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    response = requests.get(f"{NEST_API_URL}/enterprises/YOUR_PROJECT_ID/devices", headers=headers)
    devices = response.json().get('devices', [])
    events = []
    for device in devices:
        if 'sdm.devices.types.DOORBELL' in device['type']:
            events.append(device['traits']['sdm.devices.traits.DoorbellChime'])
        if 'sdm.devices.types.CAMERA' in device['type']:
            events.append(device['traits']['sdm.devices.traits.CameraPerson'])
            events.append(device['traits']['sdm.devices.traits.CameraSound'])
            events.append(device['traits']['sdm.devices.traits.CameraMotion'])
    return events


def send_firebase_notification(title, body):
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        topic='alerts',  # Ensure your app is subscribed to this topic
    )
    response = messaging.send(message)
    print('Successfully sent message:', response)


def detect_person(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)

    with torch.no_grad():
        predictions = model([image_tensor])

    return any(p['labels'].numpy() == 1 for p in predictions)  # 1 is the label for 'person'


def main():
    events = get_nest_events()
    for event in events:
        if event:
            if 'Person' in event['eventType']:
                send_firebase_notification("Person Detected", "A person has been detected by your Nest camera.")
            elif 'Sound' in event['eventType']:
                send_firebase_notification("Sound Detected", "A sound has been detected by your Nest camera.")
            elif 'Motion' in event['eventType']:
                send_firebase_notification("Motion Detected", "Motion has been detected by your Nest camera.")


if __name__ == "__main__":
    main()