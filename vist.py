import json

class VIST:
    def __init__(self, sis_file = None, dii_file= None):
        if sis_file != None:
            sis_dataset = json.load(open(sis_file, 'r'))
            self.LoadAnnotations(sis_dataset)
        else: exit('sis not found, terminating')
        
        if dii_file != None:
            dii_dataset = json.load(open(dii_file, 'r'))
            self.LoadDii(dii_dataset)
        else: exit('dii not found, terminating')


    def LoadAnnotations(self, sis_dataset = None):
        images = {}
        stories = {}

        if 'images' in sis_dataset:
            for image in sis_dataset['images']:
                images[image['id']] = image

        if 'annotations' in sis_dataset:
            annotations = sis_dataset['annotations']
            for annotation in annotations:
                story_id = annotation[0]['story_id']
                stories[story_id] = stories.get(story_id, []) + [annotation[0]]

        self.images = images # images = {imageid: imageinstance(title, tags, id... etc)}
        self.stories = stories # stories = {story_id: [storyinstance0,1,2,3,4]} each instance contain photo_flickr_id, text, original_text 
        
        #self.imagetags = self.images.values()[i]['tags']
        #self.imagetitles = self.images.values()[i]['title']
        #self.storytext = self.stories['story_id'][i]['text']
        #self.storyoriginal = self.stories['story_id'][i]['original_text']
        
    def LoadDii(self, dii_dataset):
        descs = {} 
        if 'annotations' in dii_dataset:
            annotations = dii_dataset['annotations']
            for annotation in annotations:
                imgid = annotation[0]['photo_flickr_id']
                description = annotation[0]['text']
                descs[imgid] = description
        self.descs = descs # {imgid: description string }

            # need to be written

