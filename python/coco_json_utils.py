from pathlib import Path
import json
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="COCO JSON generator")

    parser.add_argument("-md", "--mask_definition", dest="mask_definition",
    help="Path to mask_definition json file")
    parser.add_argument("-di", "--dataset_info", dest="dataset_info",
    help="Path to dataset_info json file")
    
    args = parser.parse_args()
    print(args)

    #cjc = CocoJsonCreator()
    #cjc.main(args)

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
        info['date_created'] = 
        
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

class CocoJsonCreator():
    def validate_and_process_args(self, args):
        """ Checks the arguments coming from the command line and process them
        Args:
            args: arguments parsed by argument parser
        """
        # Checking if mask_definition file exists
        mask_definition_file = Path(args.mask_definition)
        if not(mask_definition_file.exists and mask_definition_file.is_file()):
            raise FileNotFoundError(f'mask_definition.json not found: {mask_definition_file}')

        # Loading mask_definition.json
        with open(mask_definition_file) as json_file:
            self.mask_definition = json.load(json_file)
        
        # Checking if dataset_info file exists
        dataset_info_file = Path(args.dataset_info)
        if not(dataset_info_file.exists and dataset_info_file.is_file()):
            raise FileNotFoundError(f'dataset_info.json not found: {dataset_info_file}')

        # Loading dataset_info.json
        with open(dataset_info_file) as json_file:
            self.dataset_info = json.load(json_file)
        
        assert 'info' in dataset_info, 'key info missing in dataset_info.json'
        assert 'license' in dataset_info, 'key license missing in dataset_info.json'
    
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
        return lju.create_coco_license(
            url = license_json['url']
            id = license_json['id']
            name = license_json['name']
        )

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

    def main(self, args):
        self.validate_and_process_args(args)
        info = self.create_info()
        licenses = self.create_licenses()
        categories, category_id_by_name = self.create_categories()

    
