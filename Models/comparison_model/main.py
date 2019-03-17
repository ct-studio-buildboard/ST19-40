import os
import argparse
import torch
import comparison_model_train
print ("WTF")
parser = argparse.ArgumentParser(description='Neural Style Transfer')

content_path="/Users/computer/PycharmProjects/Course_Work_Cornell_Tech2018/Deep_Learning/Font_Style_Models/Neural-Style-Transfer/images/content/aerial_greek.jpg"
style_path="/Users/computer/PycharmProjects/Course_Work_Cornell_Tech2018/Deep_Learning/Font_Style_Models/Neural-Style-Transfer/images/content/smooth_a.jpg"
# Image parameters
#'images/content/neckarfront.jpg'

#print(os.cwd())
parser.add_argument('-content-image', type=str, default=content_path, help='Content image')
parser.add_argument('-style-image', type=str, default=style_path, help='Style image')
parser.add_argument('-output-image', type=str, default='output_see', help='Output image')

#Learning parameters
parser.add_argument('-weight-decay', type=float, default=0.0001, help='Weight decay to use for training')
parser.add_argument('-lr', type=float, default=0.1, help='Learning rate to use for training')
parser.add_argument('-image-size', type=int, default=256, help='Height and width of the images')
parser.add_argument('-content-loss-weight', type=float, default=1, help='Content loss weight to use for training')
parser.add_argument('-style-loss-weight', type=float, default=1000, help='Style loss weight to use for training')

#CV parameters
parser.add_argument('-iterations', type=int, default=400, help='Number of iterations for train [default: None]')
parser.add_argument('-num-models', type=int, default=50, help='Number of models [default: None]')
parser.add_argument('-save-dir', type=str, default='images/output/', help='Where to save the snapshots')

args = parser.parse_args()
if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    #print("\t{}={}".format(attr.upper(), value))
    print

comparison_model_train.train(args)


#https://github.com/animesh-s/Neural-Style-Transfer