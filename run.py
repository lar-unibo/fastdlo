from fastdlo.core import Pipeline
import os, cv2
import argparse

parser = argparse.ArgumentParser(description='FASTDLO: Fast Deformable Linear Objects Instance Segmentation and Modelling')
parser.add_argument('--img', required=True, help='source image path to process')
parser.add_argument('--img_w', default=640, type=int, help='image width')
parser.add_argument('--img_h', default=360, type=int, help='image height')
parser.add_argument('--ckpt_seg', default="CP_segmentation.pth", help='name checkpoint segmentation network')
parser.add_argument('--ckpt_siam', default="CP_similarity.pth", help='name checkpoint similarity network')
args = parser.parse_args()


if __name__ == "__main__":

    script_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint_sim = os.path.join(script_path, "weights/" + args.ckpt_siam)
    checkpoint_seg = os.path.join(script_path, "weights/" + args.ckpt_seg)

    p = Pipeline(checkpoint_siam=checkpoint_sim, checkpoint_seg=checkpoint_seg, img_w=args.img_w, img_h=args.img_h)
 
    source_img = cv2.imread(args.img, cv2.IMREAD_COLOR)
    source_img = cv2.resize(source_img, (args.img_w, args.img_h))

    img_out = p.run(source_img=source_img)

    conc = cv2.hconcat([source_img, img_out])
    cv2.imshow("output", conc)
    cv2.waitKey(0)

