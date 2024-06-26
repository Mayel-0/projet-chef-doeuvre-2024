import os 
import concurrent.futures
import face_recognition
import cv2
import numpy as np
from tqdm import tqdm
import uuid

#Live webcam recognition
def webcam_recognition(known_face_encodings, known_face_names):
    try:
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()

            rgb_frame = frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
    except:
        print("Webcam not found or not working")
        exit()

# Get all images in a directory
def get_images(path):
    try:
        images = []
        for root, dirs, files in os.walk(path):
            for file in files:
                images.append(os.path.join(root, file))
        return images
    except:
        print("Unable to get images in directory")
        exit()

def get_face(image):
    try:
        image = face_recognition.load_image_file(image)
        face_locations = face_recognition.face_locations(image)
        return face_locations[0]
    except IndexError:
        print(f"No face found in: {image}")
        exit()


def draw_rectangle(image, output, coordinates, name_file_img):
    try:
        image =  cv2.imread(image)
        x, y, w, h = coordinates
        cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 10)
        #cv2.rectangle(image, (x, y - 100), (w, y), (0, 0, 255), cv2.FILLED)
        #font = cv2.FONT_HERSHEY_DUPLEX
        #cv2.putText(image, name_file_img, (x + 6, y - 6), font, 3.5, (255, 255, 255), 3)
        cv2.imwrite(os.path.join(os.getcwd(), "output", output) ,image)
    except:
        print("Error while drawing rectangle")
        exit()

def encode_face(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        face_name = os.path.basename(image_path).replace(".jpeg", "")
        return (face_encoding, face_name)
    except IndexError:
        print(f"No face found in: {image_path}")
        exit()
    


def face_encoding():
    try:
        known_people = get_images(os.path.join(os.getcwd(), "known_people_face"))
        
        known_face_encodings = []
        known_face_names = []

        core = max(1, os.cpu_count() // 2)

        with concurrent.futures.ProcessPoolExecutor(max_workers = core) as executor:
            results = list(tqdm(executor.map(encode_face, known_people), total=len(known_people), desc="Loading faces"))
        
        for result in results:
            if result:
                known_face_encodings.append(result[0])
                known_face_names.append(result[1])
        
        return known_face_encodings, known_face_names
    except:
        print("Unable to encode faces")
        exit()

def start_image(known_face_encodings, known_face_names):
    try:
        face_locations = []
        face_encodings = []
        face_names = []

        name_file = input("Which image are you trying to load: ")
        name_file_test = f"{name_file}Vtest-{uuid.uuid4()}.jpeg"
        name_file = os.path.join(os.getcwd(), "image_test", f"{name_file}.jpeg")

        unknown_image = face_recognition.load_image_file(name_file)
        unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

        results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.55)

        print("under process...")

        print(name_file)

        print(get_face(name_file))

        draw_rectangle(name_file, name_file_test, get_face(name_file), name_file)

        print(results)

        for i, result in enumerate(results):
            if result:
                print("Known person: ", known_face_names[i])

        print("Process ended !")
    except:
        print("Unable to process images")
        exit()

def main():
    try:
        known_face_encodings, known_face_names = face_encoding()
        state = input("webcam ou image: ")
        if state == "webcam":
            webcam_recognition(known_face_encodings, known_face_names)
        if state == "image":
            start_image(known_face_encodings, known_face_names)
    except:
        print("Error occured try again webcam/image")
        exit()

if __name__ == '__main__':
    main()