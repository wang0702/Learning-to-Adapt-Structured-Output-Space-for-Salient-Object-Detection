from dataset.joint_dataset import get_loader
import argparse

parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

parser.add_argument("--data-dir-target", type=str, default='',
                    help="Path to the directory containing the target dataset.") # target的文件路径
parser.add_argument("--data-list-target", type=str, default='',
                    help="Path to the file listing the images in the target dataset.") # target list的路径
parser.add_argument("--data-dir", type=str, default='',
                    help="Path to the directory containing the source dataset.") # sourse文件的路径
parser.add_argument("--data-list", type=str, default='',
                    help="Path to the file listing the images in the source dataset.") # sourse list的路径
parser.add_argument("--batch-size", type=int, default=1,
                    help="Number of images sent to the network in one step.") # batchsize

args = parser.parse_args()

if __name__ == '__main__':
    loader = get_loader(args)

    maxheight, maxlenth, maxsize, allnum = 0, 0, 0, 0

    for i, batch in enumerate(loader):
        
        source_images, target_images = batch['sal_image'], batch['edge_image']

        maxheight = max(maxheight, source_images.size()[2], target_images.size()[2])
        maxlenth = max(maxlenth, source_images.size()[3], target_images.size()[3])
        maxsize = max(maxsize, source_images.size()[2]*source_images.size()[3], target_images.size()[2]*target_images.size()[3])
        allnum += 1

        if i % 1000 == 0:
            print('now:', i, 'maxheight:', maxheight, 'maxlenth:', maxlenth, 'size:', 'maxsize:', maxsize)

    print('all:', allnum)
    
