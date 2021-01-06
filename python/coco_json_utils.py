from pathlib import Path
import json
from PIL import Image
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from skimage import io, measure
from tqdm import tqdm

class InfoJsonUtils():
    """Creates the info object of coco.json file
    """
    def create_coco_info(self, description, version, url, year, contributor, date_created):
        """returns a dictionary of 'info' details for coco.json file
        """
        info = dict()
        info['description'] = description
        info['version'] =  version
        info['url'] = url
        info['year'] = year
        info['contributor'] = contributor
        info['date_created'] = date_created 
        
        return info

class LicenseJsonUtils():
    def create_coco_license(self, url, id, name):
        license = dict()
        license['url'] = url
        license['id'] = id
        license['name'] = name

        return license

class CategoryJsonUtils():
    def create_coco_category(self, supercategory, category_id, name):
        category = dict()
        category['supercategory'] = supercategory
        category['id'] = category_id
        category['name'] = name
    
        return  category

class ImageJsonUtils():
    """ Creates an image object for coco.json file
    """
    def create_coco_image(self, image_path, image_id, image_license):
        """ Returns:
                imaage: A dictionary of image details
        """
        # Open image and get the size
        image_file = Image.open(image_path)
        width, height = image_file.size

        image = dict()
        image['license'] = image_license
        image['file_name'] = image_path.name
        image['width'] = width
        image['height'] = height
        image['id'] = image_id

        return image

class  AnnotationJsonUtils():
    """ Creates annotations object for coco.json file
    """
    def __init__(self):
        self.annotation_id_index = 0
    
    def create_coco_annotations(self, image_mask_path, image_id, category_ids):
        """ Takes a pixel based RGB image mask(ie. the mask we made for each image) and creates COCO annotations
        Args:
            image_mask_path: Path to image mask
            image_id: Integer id of image
            category_id: A dictionary with keys: rgb color value in string , values: category id of the category in that rgb color
                        (eg: {'(255, 0, 0)': 1}): --> Means red color corresponds to a category with category id 1 in that mask
        Returns:
            annotations: a list of COCO annotation dictionaries that can
            be converted to json. e.g.:
            {
                "segmentation": [[101.79,307.32,69.75,281.11,...,100.05,309.66]],
                "area": 51241.3617,
                "iscrowd": 0,
                "image_id": 284725,
                "bbox": [68.01,134.89,433.41,174.77],
                "category_id": 6,
                "id": 165690
            }
        """
        # Set class variables
        self.image_id = image_id
        self.category_ids = category_ids

        # Make sure keys in category_ids are strings
        for key in self.category_ids.keys():
            if type(key) is not str:
                raise TypeError("category_ids keys must be string (eg: '(0, 255, 0)')")
            break

        # Open and process image
        self.mask_image = Image.open(image_mask_path)
        self.mask_image = self.mask_image.convert('RGB')
        self.width, self.height = self.mask_image.size

        # Split up of multi-coloured mask into multiple black and white colored mask
        self._isolate_masks()

        # Creating annotations from the split up masks
        self._create_annotations()

        return self.annotations

    def _isolate_masks(self):
         # Breaks up multi-coloured mask to no:of single colored masks

         self.isolated_masks = dict()
         for x in range (self.width):
             for y in range(self.height):
                 pixel_rgb = self.mask_image.getpixel((x,y))
                 pixel_rgb_str = str(pixel_rgb)

                 # If pixel is any other color other than black then add it a respective isolated mask
                 if not pixel_rgb == (0, 0, 0):
                     if self.isolated_masks.get(pixel_rgb_str) is None:
                         # Isolated mask for that rgb color doesn't exist yet,
                         # create an image with 1-bit pixels, default black. Add a padding of 1 pixel
                         # on the sides to allow the contour algorithm to work when shapes bleed
                         # to the edges
                         self.isolated_masks[pixel_rgb_str] = Image.new('1', (self.width + 2, self.height + 2))

                    # Isolated mask image already exist. So add the new pixel to the isolated mask by shifiting 1 pixel accounting the padding
                     self.isolated_masks[pixel_rgb_str].putpixel((x+1, y+1), 1)
        
    def _create_annotations(self):
        # Creates annotation for each isolated mask

        # Each image may have multiple annotations, so create an array
        self.annotations = []
        for key, mask in self.isolated_masks.items():
            annotation = dict()
            annotation['segmentation'] = []
            annotation['iscrowd'] = 0
            annotation['image_id'] = self.image_id
            if not self.category_ids.get(key):
                print(f'category color not found: {key}; Check whether color defined in mask definiton file and color in image mask are same. Image id: {self.image_id}')
                continue
            annotation['category_id'] = self.category_ids[key]
            annotation['id'] = self._next_annotation_id()

            # Find contours in isolated mask
            mask = np.asarray(mask, dtype=np.float32)
            contours = measure.find_contours(mask, 0.5, positive_orientation='low')

            # making a polygon array to store the contours as polygons
            polygons = []
            for contour in contours:
                # Contour is as (col, row) format, flip it to (x, y) format
                # and subtract the padding
                for i in range(len(contour)):
                    col, row = contour[i]
                    contour[i] = (row - 1, col - 1)

                # Make a polygon and simplify it
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)

                # Ignore tiny polygons and considering only larger ones
                if (poly.area > 16):
                    if (poly.geom_type == 'MultiPolygon'):
                        # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
                        poly = poly.convex_hull
                    if (poly.geom_type == 'Polygon'): # This check is done to ensure that it is not a line or point
                        polygons.append(poly)
                        segmentation = np.array(poly.exterior.coords).ravel().tolist()
                        annotation['segmentation'].append(segmentation)

            if len(polygons) == 0:
                # If there are no polygons then ignore it
                continue

            # Combine the polygons to calculate the bounding box and area
            multi_poly = MultiPolygon(polygons)
            x, y, max_x, max_y = multi_poly.bounds
            self.width = max_x - x
            self.height = max_y - y
            annotation['bbox'] = (x, y, self.width, self.height)
            annotation['area'] = multi_poly.area

            # Finally, add this annotation to the list
            self.annotations.append(annotation)        


    def _next_annotation_id(self):
        # Gets the next annotation id
        annotation_id = self.annotation_id_index
        self.annotation_id_index += 1
        return annotation_id

class CocoJsonCreator():
    def validate_and_process_args(self, args):
        """ Checks the arguments coming from the command line and process them
        Args:
            args: arguments parsed by argument parser
        """
        # Checking if mask_definition file exists
        mask_definition_file = Path(args.mask_definition)
        if not(mask_definition_file.exists and mask_definition_file.is_file()):
            raise FileNotFoundError(f'mask_definitions.json not found: {mask_definition_file}')

        # Loading mask_definition.json
        with open(mask_definition_file) as json_file:
            self.mask_definition = json.load(json_file)
        
        self.dataset_dir = mask_definition_file.parent

        # Checking if dataset_info file exists
        dataset_info_file = Path(args.dataset_info)
        if not(dataset_info_file.exists and dataset_info_file.is_file()):
            raise FileNotFoundError(f'dataset_info.json not found: {dataset_info_file}')

        # Loading dataset_info.json
        with open(dataset_info_file) as json_file:
            self.dataset_info = json.load(json_file)
        
        assert 'info' in self.dataset_info, 'key info missing in dataset_info.json'
        assert 'license' in self.dataset_info, 'key license missing in dataset_info.json'
    
    def create_info(self):
        """returns a dictionary of 'info' details for coco.json file
        """
        info_json = self.dataset_info['info']
        iju = InfoJsonUtils()
        return iju.create_coco_info(
            description = info_json['description'],
            version = info_json['version'],
            url = info_json['url'],
            year = info_json['year'],
            contributor = info_json['contributor'],
            date_created = info_json['date_created']
        )

    def create_license(self):
        """returns a dictionary of 'licenses' details for coco.json file
        """
        license_json = self.dataset_info['license']
        lju = LicenseJsonUtils()
        lic = lju.create_coco_license(
            url = license_json['url'],
            id = license_json['id'],
            name = license_json['name']
        )

        return [lic]

    def create_categories(self):
        """Returns:
            categories: a list of 'categories' details for coco.json file
            category_id_by_name: reference for category id based on the name of the category
        """
        cju = CategoryJsonUtils()
        categories = []
        category_id_by_name = dict()
        category_id = 1

        super_categories = self.mask_definition['super_categories']
        for super_category, _categories in super_categories.items():
            for category_name in _categories:
                categories.append(cju.create_coco_category(super_category, category_id, category_name))
                category_id_by_name[category_name] = category_id
                category_id+=1

        return categories, category_id_by_name

    def create_images_and_annotations(self, category_id_by_name):
        """ Returns:
                image_list: A list of images for coco.json file
                annotation_list: A list of annotations for coco.json file
        """
        iju = ImageJsonUtils()
        aju = AnnotationJsonUtils()

        image_objs = []
        annotation_objs = []
        image_license = self.dataset_info['license']['id']
        image_id = 0

        mask_count = len(self.mask_definition['masks'])
        print(f'Processing {mask_count} image masks...')

        # For each image mask, create images and annotations for coco.json
        for file_name, mask_def in tqdm(self.mask_definition['masks'].items()):
            # create coco image json item
            image_path = Path(self.dataset_dir) / file_name
            image_obj = iju.create_coco_image(image_path, image_id, image_license)
            image_objs.append(image_obj)

            mask_path = Path(self.dataset_dir) / mask_def['mask']

            # Create a dict of category ids where key: rgb color ('(0, 255, 0)'), value: category ids of categories in mask
            category_ids_by_rgb = dict()
            for rgb_color, category in mask_def['color_categories'].items():
                category_ids_by_rgb[rgb_color] = category_id_by_name[category['category']]

            annotation_obj = aju.create_coco_annotations(mask_path, image_id, category_ids_by_rgb)
            annotation_objs += annotation_obj
            image_id += 1

        return image_objs, annotation_objs


    def main(self, args):
        self.validate_and_process_args(args)
        info = self.create_info()
        licenses = self.create_license()
        categories, category_id_by_name = self.create_categories()
        images, annotations = self.create_images_and_annotations(category_id_by_name)

        master_obj = {
            'info': info,
            'license': licenses,
            'images': images,
            'annotations': annotations,
            'categories': categories
        }

        # Writing all info to the json file
        output_path = Path(self.dataset_dir) / 'coco.json'
        with open(output_path, 'w+') as output_file:
            json.dump(master_obj, output_file)

        print(f'Successfully created coco.json file in: \n{output_path}')
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="COCO JSON generator")

    parser.add_argument("-md", "--mask_definition", dest="mask_definition",
    help="Path to mask_definition json file")
    parser.add_argument("-di", "--dataset_info", dest="dataset_info",
    help="Path to dataset_info json file")
    
    args = parser.parse_args()
    print(args)

    cjc = CocoJsonCreator()
    cjc.main(args)