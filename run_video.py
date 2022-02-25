from fastdlo.core import Pipeline
import os, cv2, arrow
import argparse

parser = argparse.ArgumentParser(description='FASTDLO: Fast Deformable Linear Objects Instance Segmentation and Modelling')
parser.add_argument('--video', required=True, help='path to source video to process')
parser.add_argument('--img_w', default=640, help='image width')
parser.add_argument('--img_h', default=360, help='image height')
parser.add_argument('--ckpt_seg', default="CP_segmentation.pth", help='name checkpoint segmentation network')
parser.add_argument('--ckpt_siam', default="CP_similarity.pth", help='name checkpoint similarity network')
args = parser.parse_args()


if __name__ == "__main__":

    script_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint_sim = os.path.join(script_path, "weights/" + args.ckpt_siam)
    checkpoint_seg = os.path.join(script_path, "weights/" + args.ckpt_seg)

    p = Pipeline(checkpoint_siam=checkpoint_sim, checkpoint_seg=checkpoint_seg, img_w=args.img_w, img_h=args.img_h)
    cap = cv2.VideoCapture(args.video)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', args.img_w*2,args.img_h)

    if (cap.isOpened()== False): 
        print("Error opening video  file")
    
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            frame = cv2.resize(frame, (args.img_w, args.img_h))
        
            t0 = arrow.utcnow()

            img_out = p.run(source_img=frame)

            tot_time = "FPS: {0:.0f}".format(1/(arrow.utcnow() - t0).total_seconds())
            
            conc = cv2.hconcat([frame, img_out])
            
            cv2.putText(conc, tot_time, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1, 1)            
            cv2.imshow("output", conc)
            cv2.waitKey(1)
        
        else: 
            break

    cap.release()
    cv2.destroyAllWindows()
