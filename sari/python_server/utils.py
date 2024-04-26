import math
import cv2
import numpy as np


def draw_grid(image, cell_size):
    height, width, _ = image.shape
    for y in range(0, height, cell_size):
        cv2.line(image, (0, y), (width, y), (0, 0, 0), 1)
    for x in range(0, width, cell_size):
        cv2.line(image, (x, 0), (x, height), (0, 0, 0), 1)
    return image

def generate_random_colors(grid_size):
    # Generate random numbers representing color values (0 to 1)
    colors = np.random.rand(grid_size[0], grid_size[1])
    return colors

def calculate_color(grid_center, box_centers, box_sizes):
    colors = []
    for box_center, box_size in zip(box_centers.values(), box_sizes.values()):
        distance = np.linalg.norm(np.array(box_center) - np.array(grid_center))
        normalized_distance = distance / np.sqrt(box_size)
        color_value = 1 - np.clip(normalized_distance, 0, 1)  # Ensure color value is within 0 to 1
        colors.append(color_value)
    return max(colors) if colors else 0


def color_grid(image, colors, cell_size, fire_sizes, fire_centers, tree_sizes, tree_centers):
    height, width, _ = image.shape
    grid_height, grid_width = colors.shape
    color_values = {}
    for y in range(0, height, cell_size):
        for x in range(0, width, cell_size):
            center_x = int(x + grid_width/2)
            center_y = int(y + grid_height/2)
            fire_distances = {}
            for i, (x_fire, y_fire) in fire_centers.items():
                fire_distances[i] = np.sqrt((x_fire-center_x)**2 + (y_fire-center_y)**2)
            tree_distances = {}
            for i, (x_tree, y_tree) in tree_centers.items():
                tree_distances[i] = np.sqrt((x_tree-center_x)**2 + (y_tree-center_y)**2)
            for i in fire_centers.keys():
                print(f"At Cell(X, Y): ({center_x, center_y}) the distance from {i}th Fire at {fire_centers[i]} is {fire_distances[i]}.")
            for i in tree_centers.keys():
                print(f"At Cell(X, Y): ({center_x, center_y}) the distance from {i}th Tree at {tree_centers[i]} is {tree_distances[i]}.")
            # Determine the color for the current grid cell
            color_value = 0.55 * np.sum(np.array(list(fire_sizes.values()))*0.2 / 1.5*np.array(list(fire_distances.values()))) + 0.45 * np.sum(2.2*np.array(list(tree_sizes.values())) / 0.8*np.array(list(tree_distances.values())))
            color_values[y, x] = color_value

    max_color_value = max(color_values.values())
    min_color_value = min(color_values.values())
    for key in color_values:
        color_values[key] /= (max_color_value - min_color_value)
        # color_values[key] *= 100
        print(color_values[key])

    for y in range(0, height, cell_size):
        for x in range(0, width, cell_size):        
            blue_value = int(abs(1 - color_values[y, x]) * 255)  # Blue value (closer to 0)
            red_value = int(color_values[y, x] * 255)  # Red value (closer to 1)
            color = (blue_value, 0, red_value)  # Blue to Red color gradient
            # print("Color:", color)
            cv2.rectangle(image, (x, y), (x + cell_size, y + cell_size), color, -1)
    return image

def map_with_colors(path, fire_sizes, fire_centers, tree_sizes, tree_centers, cell_size=25):
    image = cv2.imread(path)
    grid_image = draw_grid(image.copy(), cell_size)
    grid_size = (grid_image.shape[0] // cell_size, grid_image.shape[1] // cell_size)
    colors = generate_random_colors(grid_size)
    colored_image = color_grid(grid_image, colors, cell_size, fire_sizes, fire_centers, tree_sizes, tree_centers)
    colored_image = cv2.resize(colored_image, (1067, 600))
    return colored_image


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

    sizes = {}
    centers = {}
    fire_boxes = [cv2.boundingRect(cnt) for cnt in fire_contours]
    for i, (x, y, w, h) in enumerate(fire_boxes):
        sizes[i] = w*h
        centers[i] = (int(x+w/2), int(y+h/2))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image, fire_boxes, sizes, centers

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
    
    sizes = {}
    centers = {}
    for i, (x, y, w, h) in enumerate(tree_boxes):
        sizes[i] = w*h
        centers[i] = (int(x+w/2), int(y+h/2))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, tree_boxes, sizes, centers

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


def detect_stuff(path):
    images=[]
    images.append(cv2.imread(path))
    # images.append(cv2.imread("./python_server/static/fire2.jpg"))
    # images.append(cv2.imread("./python_server/static/fire3.jpg"))
    # images.append(cv2.imread("./python_server/static/fire4.jpg"))
    
    hahaimages = []
    for i, image in enumerate(images):
        screen_width = 800
        screen_height = 600
        resized_image_fire = cv2.resize(image, (screen_width, screen_height))

        resized_image_tree = cv2.resize(image, (screen_width, screen_height))
        resized_image_smoke = cv2.resize(image, (screen_width, screen_height))

        image_with_fire, fire_boxes, fire_sizes, fire_centers = detect_fire(resized_image_fire)
        image_with_trees, tree_boxes, tree_sizes, tree_centers = detect_trees(resized_image_tree)
        image_with_smoke, somke_boxes = detect_smoke(resized_image_smoke)
        
        print("\nFire Sizes:", fire_sizes)
        print("\nTree Sizes:", tree_sizes)
        print("\nFire Centers:", fire_centers)
        print("\nTree Centers:", tree_centers)

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
    return hahaimages, fire_sizes, fire_centers, tree_sizes, tree_centers


if __name__ == "__main__":
    images, fire_sizes, fire_centers, tree_sizes, tree_centers = detect_stuff()
    for image in images:
        fire, smoke, tree = image[0], image[1], image[2]
        # cv2.imshow("Fire Detection", fire)
        # cv2.imshow("Tree Detection", tree)
        # cv2.imshow("Smoke Detection", smoke)
        # now i want to save the images in the directory so i can call them in another function
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    map_with_colors("./python_server/static/fire.png", fire_sizes, fire_centers, tree_sizes, tree_centers, cell_size=10)

