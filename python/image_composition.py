from pathlib import Path
import json
import warnings

class MaskJsonUtils():
    """ Creates the mask_definition.json file
    """

    def __init__(self, output_dir):
        """ Initializes the class
        Args:
            output_dir: the directory where the mask_definition file will be stored
        """
        self.output_dir = output_dir
        self.masks = dict()
        self.super_categories = dict()

    def add_category(self, category, super_category):
        """ Adds a new category to the corresponding super_category
        Args:
            True if successful, False if the category was already in the dictionary
        """

        if not self.super_categories.get(super_category):
            # Super category doesn't exist yet. Create a new set
            self.super_categories[super_category] = {category}

        elif category in self.super_categories[super_category]:
            # category already in set
            return False
        else:
            # Add the category to existing set
            self.super_categories[super_category].add(category)
        
        return True # Addition successful

    def add_mask(self, image_path, mask_path, color_categories):
        """ Takes an image path, mask path, color categories and adds it to appropriate dictionaries as shown below 

                {
                    "masks":
        # This dict{
                        "images/01.png":
                        {
                            "mask": "masks/01.png",
                            "color_categories":
                            {
                                "(255, 0, 0)": {"category": "moving_box", "super_category": "cardboard_box"},
                                "(255, 255, 0)": {"category": "moving_box", "super_category": "cardboard_box"}
                            }
                        }
                    }
                }
        Args:
            image_path: Path of image
            mask_path: Path of corresponding image mask
            color_categories: dict(rgb_color keyed dictionary having category and super category) as shown below

                            { "(0, 0, 255)": {"category": "moving_box", "super_category": "cardboard_box"} }
        Returns:
            True if successful, False if the image is already in the dictionary
        """
        if self.masks.get(image_path):
            return False # image already present in the mask_definitions

        # Image not present, create mask details for new image
        mask = {
            'mask': mask_path,
            'color_categories': color_categories
        }

        # Add new image and mask details to the mask_definition
        self.masks[image_path] = mask

        # Collecting category and super category information from the color_categories
        for _, item in color_categories.items():
            self.add_category(item['category'], item['super_category'])

        return True # Addition successfull

def get_masks(self):
    """ Gets all masks that have been added
    """
    return self.masks

def get_super_categories(self):
    """  Used to convert the sets in self.super_categories to list so that it can be serialized as JSON 
        Sets are not serializable
        Returns:
            A dictionary of super categories with value of each super category being a list of categories
            eg: { 'animal': ['dog', 'cat'], 'bird': ['parrot', 'sparrow'] }
    """
    
    serialized_super_cats = dict()
    for super_cats, category_set in self.super_categories.items():
        serialized_super_cats[super_cats] = list(category_set)
    
    return serialized_super_cats

def write_masks_to_json(self):
    """ Writes all the mask and color categories to output file path as JSON
    """
    # Serialize the mask and super categories dictionaries
    serializable_mask = self.get_masks()
    serializable_super_cats = self.get_super_categories()
    mask_obj = {
        'masks': serializable_mask,
        'super_categories': serializable_super_cats
    }

    # Write the mask_obj to JSON file
    output_file_path = Path(self.output_dir) / 'mask_definitions.json'
    with open(output_file_path, 'w+') as json_file:
        json_file.write(json.dump(mask_obj))

class ImageComposition():
    """Composes images together in random ways, applying transformations to the foreground to create a synthetic combined image
    """  

    def __init__(self):
        self.allowed_output_types = ['.png', '.jpg', '.jpeg']
        self.allowed_background_types = ['.png', '.jpg', '.jpeg']
        self.zero_padding = 8 # 00000027.png, supports up to 100 million images
        self.max_foregrounds = 3
        self.mask_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        assert len(self.mask_colors) >= self.max_foregrounds, 'length of mask_colors should be >= max_foregrounds'

    def _validate_and_process_args(self, args):
        """ Validates input arguments and sets up class variables
        Args:
            args: the arguments from commandline
        """

        self.silent = args.silent

        # Validate count
        assert args.count > 0, 'count must be greater than 0'
        self.count = args.count

        # Validate width and height
        assert args.width >= 64, 'width must be greater than 64px'
        self.width = args.width
        assert args.height >= 64, 'height must be greater than 64px'
        self.height = args.height

        # Validate and process output type
        if args.output_type is None:
            self.output_type = '.jpg' # default
        else:
            if args.output_type[0] != '.':
                self.output_type = f'.{args.output_type}'
            else:
                self.output_type = args.output_type
            
            assert self.output_type in self.allowed_output_types, f'output type not supported: {self.output_type}'

        # Validate and process output and input directory
        self.validate_and_process_output_directory()
        self.validate_and_process_input_directory()

    def validate_and_process_output_directory(self):
        self.output_dir = Path(args.output_dir)
        self.images_output_dir = self.output_dir / 'images'
        self.mask_output_dir = self.output_dir / 'masks'

        # Create directories
        self.output_dir.mkdir(exist_ok=True) # exist_ok=True means its okay if the directory is already there
        self.images_output_dir.mkdir(exist_ok=True)
        self.mask_output_dir.mkdir(exist_ok=True)

        if not self.silent:
            # Check for existing contents in image directory
            for _ in self.images_output_dir.iterdir():
                # We found something, check if the user wants to overwrite the files or quit
                should_continue = input('output directory is not empty, files may be overwritten, \n Do you wish to continue(y/n)').lower()
                if should_continue != 'y' and should_continue != 'yes':
                    quit()
                break

    def validate_and_process_input_directory(self):
        self.input_dir = Path(args.input_dir)
        assert self.input_dir.exists(), f'Input directory does not exist: {self.input_dir}'

        for x in self.input_dir.iterdir():
            if x.name == 'foregrounds':
                self.foregrounds_dir = x
            elif x.name == 'backgrounds':
                self.backgrounds_dir = x

        assert self.foregrounds_dir is not None, f'foregrounds folder was not found in input directory: {self.input_dir}'
        assert self.backgrounds_dir is not None, f'backgrounds folder was not found in input directory: {self.input_dir}'

        self.validate_and_process_foregrounds()
        self.validate_and_process_backgrounds()

    def validate_and_process_foregrounds(self):
        """ Validates the input foregrounds and process them
            Expected directory structure
            + foregrounds_dir
                + super_category_dir
                    + category_dir
                        + foreground_img.png
            + backgrounds_dir
                + background_img.png
        """

        self.foregrounds_dict = dict()
         
        for super_category_dir in self.foregrounds_dir.iterdir():
            # if not directory raise warning
            if not super_category_dir.is_dir():
                warnings.warn(f'file found in foregrounds directory (expected a super categories directory), ignoring: {super_category_dir}')
                continue

            # This is a super category directory
            for category_dir in super_category_dir.iterdir():
                # if not directory raise warning
                if not category_dir.is_dir():
                    warnings.warn(f'file found in super category directory (expected a category directory), ignoring: {category_dir}')
                    continue

                # This is a category directory
                for image_file in category_dir.iterdir():
                    # if not image raise warning
                    if not image_file.is_file():
                        warnings.warn(f'directory found inside category directory (expected an image file), ignoring: {str(image_file)}')
                        continue
                    if image_file.suffix != '.png':
                        warnings.warn(f'foreground image must be a png file, skipping: {str(image_file)}')
                        continue

                    # Valid foreground image add it to foregrounds
                    super_category = super_category_dir.name
                    category = category_dir.name

                    if super_category not in self.foregrounds_dict:
                        self.foregrounds_dict[super_category] = dict()
                    
                    if category not in self.foregrounds_dict[super_category]:
                        self.foregrounds_dict[super_category][category] = []

                    self.foregrounds_dict[super_category][category].append(image_file)

        assert len(self.foregrounds_dict) > 0, 'no valid foreground images were found'

    def validate_and_process_backgrounds(self):
        self.backgrounds = []
        for image_file in self.backgrounds_dir.iterdir():
            # if not file raise warning
            if not image_file.is_file():
                warnings.warn(f'directory found inside backgrounds folder (expecting an image file), ignoring: {image_file}')
                continue

            if image_file.suffix not in self.allowed_background_types:
                warnings.warn(f'background type {image_file.suffix} not allowed. Check the list of allowable background types, ignoring: {image_file}')
                continue

            # Valid file, add to backgrounds list
            self.backgrounds.append(image_file)

        assert len(self.backgrounds) > 0, 'no valid backgrounds found'
        

    def main(self, args):
        self._validate_and_process_args(args)
        self._generate_images()
        self._create_info()
        print("Dataset creation completed")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Composition")
    parser.add_argument("--input_dir", type=str, dest="input_dir", required=True, help="Input directory. \
        It contains a 'backgrounds' folder and 'foregrounds' folder. The foreground folder contains \
        the folders listing all the super categories and each super category folder contains \
        a number of subfolders for categories. The categories folder contains png images of \
        items in that category")
    parser.add_argument("--output_dir", type=str, dest="output_dir", required=True, help=" The directory where images, masks, \
        and json files will be stored")
    parser.add_argument("--count", type=int, dest="count", required=True, help="number of images to generate")
    parser.add_argument("--width", type=int, dest='width', required=True, help=" output image pixel width")
    parser.add_argument("--height", type=int, dest="int", required=True, help="output image pixel height")
    parser.add_argument("--output_type", type=str, dest="output_type", required=True, help="png or jpg(default)")
    parser.add_argument("--silent", action='store_true', help="silent mode; doesn't prompt the user for input, \
        automatically overwrites files")

    args = parser.parse_args()
    image_comp = ImageComposition()
    image_comp.main(args)
        