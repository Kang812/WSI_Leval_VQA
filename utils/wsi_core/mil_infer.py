class tumor_detect_dataset(Dataset):
    def __init__(self, image_path, patch_size, stride, transform = None):
        self.image = cv2.imread(image_path)
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.start_coords =  self._get_path_coord(self.height, self.width, self.patch_size, self.stride)
        
    def _get_path_coord(self, height, width, patch_size, stride):
        start_coords = []
        for h in range(0, height, patch_size - stride):
            for w in range(0, width, patch_size - stride):
                start_w = w
                end_w = w + 300
                
                if end_w > width:
                    end_w = width
                    start_w = end_w - 300

                start_h = h
                end_h = h + 300

                if end_h > height:
                    end_h = height
                    start_h = end_h - 300
                
                start_coords.append([start_w, start_h])
        return start_coords
    
    def __len__(self):
        return len(self.start_coords) 
    
    def __image_size__(self):
        return self.height, self.width
    
    def __getitem__(self, index):
        start_coord = self.start_coords[index]
        crop_img = self.image[start_coord[1]:start_coord[1]+300, start_coord[0]:start_coord[0] + 300, :]
        if self.transform is not None:
            crop_img = self.transform(image = crop_img)['image']
        return crop_img