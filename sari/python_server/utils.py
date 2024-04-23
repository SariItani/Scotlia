def process_image(path, flag=False):
    if not flag:
        print("image:", path)
        return path
    else:
        # receive the image from the arduino once the flag is enabled 
        return # the image


def map(path):
    # open CV will detect fire
    # we will also detect green biomass
    # we will also detect human property like houses
    # then we will create our heatmap as so:
    # the bounding boxes of the houses have high priority if there is a fire coming towards it.
    # if there is fire inside then the box then the entire area is highest priority
    # biomass near fire are at risk of spreading more fire so the bounding box facing the fire is high priority
    # if there is fire inside the biomass box then the fire box is highest priority and it decreases radially but with a multiplier since it is inside a biomass region
    pass
