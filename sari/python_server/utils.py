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
    return np.random.rand(grid_size[0], grid_size[1])

def calculate_color_priority(tree_sizes, tree_distances, fire_sizes, fire_distances, tree_factor, fire_factor):
    if not tree_sizes or not fire_sizes:
        return 0
    
    tree_priority = sum([size / distance for size, distance in zip(tree_sizes, tree_distances)]) * tree_factor
    fire_priority = sum([size / distance for size, distance in zip(fire_sizes, fire_distances)]) * fire_factor
    
    total_factor = tree_factor + fire_factor
    color_priority = (tree_priority + fire_priority) / total_factor
    
    return color_priority

def color_grid(image, colors, cell_size, tree_boxes, fire_boxes):
    height, width, _ = image.shape
    for y in range(0, height, cell_size):
        for x in range(0, width, cell_size):
            color_value = colors[(y-1) // cell_size, (x-1) // cell_size]
            blue_value = int(color_value * 255)  # Use original color_value
            red_value = int((1 - color_value) * 255)  # Use original color_value
            color = (blue_value, 0, red_value)
            print(f"RBG: ({red_value}, 0, {blue_value}) at Pixel: ({x + cell_size // 2},{y + cell_size // 2})")

            tree_sizes, tree_distances = [], []
            for tree_box in tree_boxes:
                tree_center = (tree_box[0] + tree_box[2] // 2, tree_box[1] + tree_box[3] // 2)
                tree_distance = np.sqrt((tree_center[0] - x - cell_size // 2) ** 2 +
                                         (tree_center[1] - y - cell_size // 2) ** 2)
                tree_sizes.append(tree_box[2] * tree_box[3])
                tree_distances.append(tree_distance)

            fire_sizes, fire_distances = [], []
            for fire_box in fire_boxes:
                fire_center = (fire_box[0] + fire_box[2] // 2, fire_box[1] + fire_box[3] // 2)
                fire_distance = np.sqrt((fire_center[0] - x - cell_size // 2) ** 2 +
                                         (fire_center[1] - y - cell_size // 2) ** 2)
                fire_sizes.append(fire_box[2] * fire_box[3])
                fire_distances.append(fire_distance)

            color_priority = calculate_color_priority(tree_sizes, tree_distances, fire_sizes, fire_distances, 40, 60)
            color_value = color_priority

            blue_value = int((1 - color_value) * 255)
            red_value = int(color_value * 255)
            color = (blue_value, 0, red_value)

            cv2.rectangle(image, (x, y), (x + cell_size, y + cell_size), color, -1)
    return image

def map_with_colors(path, cell_size=25):
    image = cv2.imread(path)
    grid_image = draw_grid(image.copy(), cell_size)
    grid_size = (image.shape[0] // cell_size, image.shape[1] // cell_size)
    colors = np.zeros(grid_size)

    _, fire_boxes = detect_fire(image)
    _, tree_boxes = detect_trees(image)

    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            grid_cell_center = (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)

            tree_sizes, tree_distances = [], []
            for tree_box in tree_boxes:
                tree_center = (tree_box[0] + tree_box[2] // 2, tree_box[1] + tree_box[3] // 2)
                tree_distance = np.sqrt((tree_center[0] - grid_cell_center[0]) ** 2 +
                                         (tree_center[1] - grid_cell_center[1]) ** 2)
                tree_sizes.append(tree_box[2] * tree_box[3])
                tree_distances.append(tree_distance)

            fire_sizes, fire_distances = [], []
            for fire_box in fire_boxes:
                fire_center = (fire_box[0] + fire_box[2] // 2, fire_box[1] + fire_box[3] // 2)
                fire_distance = np.sqrt((fire_center[0] - grid_cell_center[0]) ** 2 +
                                         (fire_center[1] - grid_cell_center[1]) ** 2)
                fire_sizes.append(fire_box[2] * fire_box[3])
                fire_distances.append(fire_distance)

            color_priority = calculate_color_priority(tree_sizes, tree_distances, fire_sizes, fire_distances, 0.04, 0.1)
            colors[y, x] = color_priority

    colored_image = color_grid(grid_image, colors, cell_size, tree_boxes, fire_boxes)
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
    lower_smoke = np.array([75, 75, 75])
    upper_smoke = np.array([255, 255, 255])
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
    images = [cv2.imread("./python_server/static/fire.jpg"),
              cv2.imread("./python_server/static/fire2.jpg"),
              cv2.imread("./python_server/static/fire3.jpg")]
    hahaimages = []
    for i, image in enumerate(images):
        screen_width, screen_height = 800, 600
        resized_image_fire = cv2.resize(image, (screen_width, screen_height))
        resized_image_tree = cv2.resize(image, (screen_width, screen_height))
        resized_image_smoke = cv2.resize(image, (screen_width, screen_height))
        image_with_fire, fire_boxes = detect_fire(resized_image_fire)
        image_with_trees, tree_boxes = detect_trees(resized_image_tree)
        image_with_smoke, smoke_boxes = detect_smoke(resized_image_smoke)
        hahaimages.append([image_with_fire, image_with_smoke, image_with_trees])
        cv2.imwrite(f"./python_server/static/firedet{i}.jpg", image_with_fire)
        cv2.imwrite(f"./python_server/static/smokedet{i}.jpg", image_with_smoke)
        cv2.imwrite(f"./python_server/static/treedet{i}.jpg", image_with_trees)
        with open(f'./python_server/static/firedat{i}.txt', 'w') as file:
            file.write(str(fire_boxes))
        with open(f'./python_server/static/treedat{i}.txt', 'w') as file:
            file.write(str(tree_boxes))
        with open(f'./python_server/static/smokedat{i}.txt', 'w') as file:
            file.write(str(smoke_boxes))
    return hahaimages


if __name__ == "__main__":
    images = detect_stuff()
    for fire, smoke, tree in images:
        pass  # Do something with the processed images if needed
    map_with_colors("./python_server/static/fire.jpg", cell_size=100)