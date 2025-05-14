import math

# Function to calculate Euclidean distance between centers of two blocks
def distance_to_figure_coords(text_block_x1, text_block_y1, text_block_x2, text_block_y2, image_block_x1, image_block_y1, image_block_x2, image_block_y2):
    text_block_center_x = (text_block_x1 + text_block_x2) / 2
    text_block_center_y = (text_block_y1 + text_block_y2) / 2
    image_block_center_x = (image_block_x1 + image_block_x2) / 2
    image_block_center_y = (image_block_y1 + image_block_y2) / 2
    return math.sqrt((text_block_center_x - image_block_center_x) ** 2 + (text_block_center_y - image_block_center_y) ** 2)

# Function to check if two blocks overlap
def is_overlapping_coords(block1_x1, block1_y1, block1_x2, block1_y2, block2_x1, block2_y1, block2_x2, block2_y2):
    return not (
        block1_x2 < block2_x1
        or block1_x1 > block2_x2
        or block1_y2 < block2_y1
        or block1_y1 > block2_y2
    )

def is_overlapping(text_block, image_block):
    return is_overlapping_coords(text_block.x_1, text_block.y_1, text_block.x_2, text_block.y_2, image_block.x_1, image_block.y_1, image_block.x_2, image_block.y_2)
    
# Function to check if one block is inside another
def is_inside_coords(block1_x1, block1_y1, block1_x2, block1_y2, block2_x1, block2_y1, block2_x2, block2_y2):
    return (
        block1_x1 >= block2_x1
        and block1_x2 <= block2_x2
        and block1_y1 >= block2_y1
        and block1_y2 <= block2_y2
    )
