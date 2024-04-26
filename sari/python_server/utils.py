def process_image(path, flag=False):
    if not flag:
        print("image:", path)
        return path
    else:
        # receive the image from the arduino once the flag is enabled 
        return # the image

def draw_grid(image, cell_size):
    height, width, _ = image.shape
    for y in range(0, height, cell_size):
        cv2.line(image, (0, y), (width, y), (0, 0, 0), 1)  # Draw horizontal grid lines
    for x in range(0, width, cell_size):
        cv2.line(image, (x, 0), (x, height), (0, 0, 0), 1)  # Draw vertical grid lines
    return image

def map(path, cell_size=25):
    image = cv2.imread(path)
    grid_image = draw_grid(image.copy(), cell_size)
    grid_image = cv2.resize(grid_image, (1067, 600))
    cv2.imshow("Grid Image", grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


import cv2
import numpy as np

def detect_fire(image):
    lower_fire = np.array([0, 50, 75])
    upper_fire = np.array([96, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

    fire_mask = cv2.erode(fire_mask, None, iterations=2)
    fire_mask = cv2.dilate(fire_mask, None, iterations=4)

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 100
    fire_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    fire_boxes = [cv2.boundingRect(cnt) for cnt in fire_contours]
    for x, y, w, h in fire_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image, fire_boxes

def detect_trees(image):
    lower_trees = np.array([0, 10, 0])
    upper_trees = np.array([96, 255, 126])


    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    tree_mask = cv2.inRange(hsv, lower_trees, upper_trees)

    tree_mask = cv2.erode(tree_mask, None, iterations=3)
    tree_mask = cv2.dilate(tree_mask, None, iterations=2)

    contours, _ = cv2.findContours(tree_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 100
    tree_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    tree_boxes = [cv2.boundingRect(cnt) for cnt in tree_contours]
    for x, y, w, h in tree_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, tree_boxes


def detect_smoke(image):
    upper_smoke = np.array([255, 255, 255])
    lower_smoke= np.array([75, 75, 75])


    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

    smoke_mask = cv2.erode(smoke_mask, None, iterations=4)
    smoke_mask = cv2.dilate(smoke_mask, None, iterations=8)

    contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 100
    smoke_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    smoke_boxes = [cv2.boundingRect(cnt) for cnt in smoke_contours]
    for x, y, w, h in smoke_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (128, 128, 128), 2)

    return image, smoke_boxes

def detect_stuff():
    images=[]
    images.append(cv2.imread("./python_server/static/fire.jpg"))
    images.append(cv2.imread("./python_server/static/fire2.jpg"))
    images.append(cv2.imread("./python_server/static/fire3.jpg"))
    # images.append(cv2.imread("./python_server/static/fire4.jpg"))
    
    hahaimages = []
    for i, image in enumerate(images):
        screen_width = 800
        screen_height = 600
        resized_image_fire = cv2.resize(image, (screen_width, screen_height))
        resized_image_tree = cv2.resize(image, (screen_width, screen_height))
        resized_image_smoke = cv2.resize(image, (screen_width, screen_height))

        image_with_fire, fire_boxes = detect_fire(resized_image_fire)
        image_with_trees, tree_boxes = detect_trees(resized_image_tree)
        image_with_smoke, somke_boxes = detect_smoke(resized_image_smoke)
      
        hahaimages.append([image_with_fire, image_with_smoke, image_with_trees])

        # cv2.imshow("Fire Detection", image_with_fire)
        # cv2.imshow("Tree Detection", image_with_trees)
        # cv2.imshow("Smoke Detection", image_with_smoke)
        # # now i want to save the images in the directory so i can call them in another function
        cv2.imwrite(f"./python_server/static/firedet{i}.jpg", image_with_fire)
        cv2.imwrite(f"./python_server/static/smokedet{i}.jpg", image_with_smoke)
        cv2.imwrite(f"./python_server/static/treedet{i}.jpg", image_with_trees)
        with open (f'./python_server/static/firedat{i}.txt', 'w') as file:
            file.write(str(fire_boxes))
        with open (f'./python_server/static/treedat{i}.txt', 'w') as file:
            file.write(str(tree_boxes))
        with open (f'./python_server/static/smokedat{i}.txt', 'w') as file:
            file.write(str(somke_boxes))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return hahaimages


if __name__ == "__main__":
    images = detect_stuff()
    for image in images:
        fire, smoke, tree = image[0], image[1], image[2]
        cv2.imshow("Fire Detection", fire)
        cv2.imshow("Tree Detection", tree)
        cv2.imshow("Smoke Detection", smoke)
        # now i want to save the images in the directory so i can call them in another function
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    map("./python_server/static/fire3.jpg")