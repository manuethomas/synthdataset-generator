from pathlib import Path
import json
import warnings
from tqdm import tqdm
import random
from PIL import Image, ImageEnhance
import numpy as np
from datetime import datetime

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
            json_file.write(json.dumps(mask_obj))

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
        
    def _generate_images(self):
        # Generates a number of images and its segmentation mask, then
        # saves a mask_definition.json file that describes the dataset

        print(f'Generating {self.count} images with masks....')

        mju = MaskJsonUtils(self.output_dir)

        # Creating images and mask
        for i in tqdm(range(self.count)):
            # Randomly choose a background
            background_path = random.choice(self.backgrounds)

            num_foregrounds = random.randint(1, self.max_foregrounds)

            foregrounds = []
            for fg_i in range(num_foregrounds):
                # Randomly choose a foreground
                super_category = random.choice(list(self.foregrounds_dict.keys()))
                category = random.choice(list(self.foregrounds_dict[super_category].keys()))
                foreground_path = random.choice(self.foregrounds_dict[super_category][category])

                # Get the color
                mask_rgb_color = self.mask_colors[fg_i]

                # Adding each forground details to foregrounds list as a dict
                foregrounds.append({
                    'super_category': super_category,
                    'category': category,
                    'foreground_path': foreground_path,
                    'mask_rgb_color': mask_rgb_color
                })

            # Compose foreground and background
            composite, mask = self._compose_images(foregrounds, background_path)

            # Create filename (used for both the composite and the mask)
            save_filename = f'{ i:0{self.zero_padding}}' # eg: 00000010

            # save composite image to image sub directory
            composite_filename = f'{save_filename}{self.output_type}' # eg: 00000010.png
            composite_path = self.output_dir / 'images' / composite_filename # eg: output_dir/images/filename
            # Remove any alpha in the composite image
            composite = composite.convert('RGB')
            composite.save(composite_path)

            # save the mask image to masks sub directory
            mask_filename = f'{save_filename}.png' # masks are always png to avoid lossy compression
            mask_path = self.output_dir / 'masks' / mask_filename # eg:output_dir/masks/filename
            mask.save(mask_path)

            # setup color_categories portion for the mask
            color_categories = dict()
            for fg in foregrounds:
                # Add category and color info
                mju.add_category(fg['category'], fg['super_category'])
                color_categories[str(fg['mask_rgb_color'])] = \
                    {
                        'category': fg['category'],
                        'super_category': fg['super_category']
                    }
            
            # Add the mask to MaskJsonUtils
            mju.add_mask(
                composite_path.relative_to(self.output_dir).as_posix(),
                mask_path.relative_to(self.output_dir).as_posix(),
                color_categories
            )

        # write masks to json
        mju.write_masks_to_json()

    def _compose_images(self, foregrounds, background_path):
        # Composes a foreground image and a background image and creates a segmentation mask
        # using the specified color. Validation should already be done by now.
        # Args:
        #     foregrounds: a list of dicts with format:
        #       [{
        #           'super_category':super_category,
        #           'category':category,
        #           'foreground_path':foreground_path,
        #           'mask_rgb_color':mask_rgb_color
        #       },...]
        #     background_path: the path to a valid background image
        # Returns:
        #     composite: the composed image
        #     mask: the mask image

        # Open background and convert it to RGBA
        background = Image.open(background_path)
        background = background.convert('RGBA')

        # Crop background to desired size (self.width x self.height), randomly positioned
        bg_width, bg_height = background.size
        max_crop_x_pos = bg_width - self.width
        max_crop_y_pos = bg_height - self.height
        assert max_crop_x_pos >= 0, f'desired width, {self.width}, is greater than the background width, {bg_width}, for {str(background_path)}'
        assert max_crop_y_pos >= 0, f'desired height, {self.height}, is greater than the background height, {bg_height}, for {str(background_path)}'
        crop_x_pos = random.randint(0, max_crop_x_pos)
        crop_y_pos = random.randint(0, max_crop_y_pos)
        composite = background.crop((crop_x_pos, crop_y_pos, crop_x_pos + self.width, crop_y_pos + self.height)) # top left corner and bottom right corner points of rect given
        composite_mask = Image.new('RGB', composite.size, 0) # 0 indicates black

        # Taking foreground images from foregrounds list
        for fg in foregrounds:
            fg_path = fg['foreground_path']

            # performing transformations
            fg_image = self._transform_foreground(fg, fg_path)

            # choose random x,y positions for foreground
            max_x_position = composite.size[0] - fg_image.size[0]
            max_y_positon = composite.size[1] - fg_image.size[1]
            assert max_x_position >= 0 and max_y_positon >= 0, \
                f'foreground {fg_path} is too big for the requested output image size({self.width} x {self.height})'
            paste_position = (random.randint(0, max_x_position), random.randint(0, max_y_positon))

            # Inorder to paste the forground onto the background,we use the Image.composite method in python. 
            # It has three params. Foreground, background and mask(also an image) 
            # which specify what all contents of the foreground should be made visible. All images 
            # should be of same size.

            # Creating a new black image having same size as that of the composite(background) image and pasting the foreground image onto it
            new_fg_image = Image.new('RGBA', composite.size, color = (0, 0, 0, 0)) # fourth value represents alpha. Note: (0, 0, 0) represents black
            new_fg_image.paste(fg_image, paste_position)

            # Extract alpha details(visibility) from foreground image and paste it onto a new black image having the size of composite image
            alpha_mask = fg_image.getchannel(3) # 3 stands for A in RGBA
            new_alpha_mask = Image.new('L', composite.size, color=0) # 0-black, 255-white
            new_alpha_mask.paste(alpha_mask, paste_position)
            
            # Composite the foreground and background images
            composite = Image.composite(new_fg_image, composite, new_alpha_mask)

            # To create the composite mask

            # Grab the alpha pixels above a specific threshold
            # This is because in certain cases the edges in the new_alpha_mask might not be clearly visible (eg: grass edges).
            # And for us to create a composite mask we need to set a threshold for alpha
            alpha_threshold = 200
            mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold)) # Taking only pixels with alpha value > 200
            uint8_mask = np.uint8(mask_arr) # Converting to uint8 format which contains only 1s and 0s

            # Multiply the mask value (1 or 0) by the color in each RGB channel and combine to get the RGB mask
            mask_rgb_color = fg['mask_rgb_color']
            red_channel = uint8_mask * mask_rgb_color[0]
            green_channel = uint8_mask * mask_rgb_color[1]
            blue_channel = uint8_mask * mask_rgb_color[2]

            # Stack all these channels into one array
            rgb_mask_arr = np.dstack((red_channel, green_channel, blue_channel))
            isolated_mask = Image.fromarray(rgb_mask_arr, 'RGB')

            # create the mask for compositing with composite_mask
            isolated_alpha = Image.fromarray(uint8_mask * 255, 'L')

            # compositing foreground(isolated_mask) and background(composite_mask)
            composite_mask = Image.composite(isolated_mask, composite_mask, isolated_alpha)

        return composite, composite_mask

    def _transform_foreground(self, fg, fg_path):
        # Open foreground and get alpha channel
        # This is done to check for transparent background in foreground
        fg_image = Image.open(fg_path)
        fg_alpha = np.array(fg_image.getchannel(3))
        assert np.any(fg_alpha == 0), f'foreground needs to have transparent background: {str(fg_path)}'

        # Applying transformations

        # Rotating foreground
        angle_degrees = random.randint(0, 359)
        fg_image = fg_image.rotate(angle_degrees, resample=Image.BICUBIC, expand=True)

        # Scaling foreground
        scale = random.random() * .5 + .5 # Pick number between .5 and 1
        new_size = (int(fg_image.size[0] * scale), int(fg_image.size[1] * scale))
        fg_image = fg_image.resize(new_size, resample=Image.BICUBIC)

        # Adjust foreground brightness
        brightness_factor = random.random() * .4 + .7 # pick number between .4 and 1.1
        enhancer = ImageEnhance.Brightness(fg_image)
        fg_image = enhancer.enhance(brightness_factor)

        # Add other transformations here

        return fg_image

    def _create_info(self):
        # A convenience wizard for automatically creating dataset info
        # The user can always modify the resulting .json manually if needed

        if self.silent:
            # No user wizard in silent mode
            return

        should_continue = input('Would you like to create dataset_info json? (y/n) ').lower()
        if should_continue != 'y' and should_continue != 'yes':
            print('No problem. You can always create the json manually.')
            quit()

        print('Note: you can always modify the json manually if you need to update this.')
        info = dict()
        info['description'] = input('Description: ')
        info['url'] = input('URL: ')
        info['version'] = input('Version: ')
        info['contributor'] = input('Contributor: ')
        now = datetime.now()
        info['year'] = now.year
        info['date_created'] = f'{now.month:0{2}}/{now.day:0{2}}/{now.year}'

        image_license = dict()
        image_license['id'] = 0

        should_add_license = input('Add an image license? (y/n) ').lower()
        if should_add_license != 'y' and should_add_license != 'yes':
            image_license['url'] = ''
            image_license['name'] = 'None'
        else:
            image_license['name'] = input('License name: ')
            image_license['url'] = input('License URL: ')

        dataset_info = dict()
        dataset_info['info'] = info
        dataset_info['license'] = image_license

        # Write the JSON output file
        output_file_path = Path(self.output_dir) / 'dataset_info.json'
        with open(output_file_path, 'w+') as json_file:
            json_file.write(json.dumps(dataset_info))

        print('Successfully created {output_file_path}')

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
    parser.add_argument("--height", type=int, dest="height", required=True, help="output image pixel height")
    parser.add_argument("--output_type", type=str, dest="output_type", help="png or jpg(default)")
    parser.add_argument("--silent", action='store_true', help="silent mode; doesn't prompt the user for input, \
        automatically overwrites files")

    args = parser.parse_args()
    image_comp = ImageComposition()
    image_comp.main(args)
        