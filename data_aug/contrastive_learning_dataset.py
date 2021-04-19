from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torchvision.datasets import ImageFolder 
from data_aug.make_dataset import Medical_DB, Proposed_Medical_DB

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        if name =='simclr':
            valid_datasets = Medical_DB(transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(112),
                                                                  n_views))
        elif name =='proposed_simclr':
            valid_datasets = Proposed_Medical_DB(transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(112),2), 
                                    negative_transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(112),2))
        
        try:
            dataset_fn = valid_datasets
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn
