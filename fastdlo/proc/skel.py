import numpy as np
import cv2
from skimage.morphology import skeletonize
import arrow
import scipy.spatial.distance
import matplotlib.pyplot as plt 

class Processing():

    def __init__(self, mask, density=10, drop_endpoints=False, show_output=False):

        self.drop_endpoints = drop_endpoints
        self.kernel_size = 3
        
        skeleton = skeletonize(mask, method="lee")

        dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

        # black out borders
        skeleton[:self.kernel_size, :] = 0
        skeleton[-(self.kernel_size+1):, :] = 0
        skeleton[:, :self.kernel_size] = 0
        skeleton[:, -(self.kernel_size+1):] = 0

        endpoints = self.skeletonEndpoints(skeleton)
        intpoints = self.skeletonIntpoints(skeleton)

        int_points = list(zip(intpoints[0], intpoints[1]))
        int_points_f = self.simplifyIntersections(int_points, threshold=self.kernel_size)

        skeleton_f = skeleton.copy()
        skeleton_f[tuple([intpoints[0], intpoints[1]])] = 0
        skeleton_f[tuple([endpoints[0], endpoints[1]])] = 0


        num_labels, labels_im = cv2.connectedComponents(skeleton_f)
        #print("num labels: ", num_labels)
        #self.showLabelsCC(labels_im)

        int_points_f = self.associateLabelsToIntersections(labels_im, int_points_f)

        if True and len(int_points_f) > 0:
            points = [v["point"] for v in list(int_points_f.values())]
            int_points_f_values = np.array(points).astype(int)
            dist_int_values = dist_img[int_points_f_values[:,0], int_points_f_values[:,1]]
            for i, pos in enumerate(int_points_f_values):
                cv2.circle(skeleton_f, tuple([pos[1], pos[0]]), int(dist_int_values[i]), 0, -1)

        #print("int_points_f: \n", int_points_f)

        paths = self.pathsFromCC(num_labels, labels_im, skeleton_f)

        int_points_f = self.simplifyIntersectionsDistImg(paths, int_points_f, dist_img)
        
        #print("Processing done!")

        self.paths = paths
        self.int_points = int_points_f
        self.dist_img = dist_img

        if show_output:
            cv2.namedWindow('skeleton', 0)
            cv2.imshow('skeleton', skeleton * 255)
            cv2.namedWindow('skeleton_filtered', 0)
            cv2.imshow('skeleton_filtered', skeleton_f * 255)
            cv2.waitKey()


    def simplifyIntersectionsDistImg(self, paths, int_points, dist_img):
        if len(int_points) == 0:
            return int_points

        segmentsw = [v["segments"] for k,v in int_points.items()]
        segments = [item for sublist in segmentsw for item in sublist]
        segments = np.unique(segments)
        path_keys = list(paths.keys())
        diff = set(segments).difference(set(path_keys))

        to_delete = []
        new_int_points = {}
        for dkey in diff:
            new_segments = []
            x, y = [], []
            keys_to_delete = []
            for k,v in int_points.items():
                if dkey in v["segments"]:
                    new_segments.extend(v["segments"])
                    x.append(v["point"][0])
                    y.append(v["point"][1])
                    keys_to_delete.append(k)
            
            segments = np.unique(new_segments)
            segments = [s for s in segments if s != dkey]
            point = tuple([np.mean(x), np.mean(y)])
            new_int_points[keys_to_delete[0]] = {"point": point, "segments": segments}
            to_delete.extend(keys_to_delete)

        for d in to_delete:
            if d in int_points:
                del int_points[d]
        new_int_points.update(int_points)

        return new_int_points

    def simplifyIntersections(self, int_points, threshold):
        #print("intersection points raw: \n", int_points)
        int_points_f = {}
        inc_key = 0
        for p in int_points:
            already_inside = False
            for k, v in int_points_f.items():
                if self.distance2D(p, v) <= threshold:  
                    already_inside = True
            if not already_inside:
                int_points_f[inc_key] = p
                inc_key += 1

        #print("Simplified intsersection points: \n", int_points_f)  
        return int_points_f              

    def distance2D(self, point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


    def associateLabelsToIntersections(self, labels_im, int_points_f):
        for k, point in int_points_f.items():
            x = point[0]
            y = point[1]
            x_range = (x-self.kernel_size*2, x+self.kernel_size*2)
            y_range = (y-self.kernel_size*2, y+self.kernel_size*2)

            label_crop = labels_im[x_range[0]:x_range[1], y_range[0]:y_range[1]]
            int_points_f[k] = {"point": point, "segments": [v for v in np.unique(label_crop) if v != 0]}

            #self.showLabelsCC(label_crop)

        #print("Simplified intsersection points with segments data: \n", int_points_f)  
        return int_points_f
        

    def showLabelsCC(self, labels):

        unique_labels = np.unique(labels)
        #print(unique_labels)
        for l in unique_labels:
            #print(l)
            mask = np.zeros_like(labels)
            mask[labels == l] = 255
            mask = mask.astype(np.uint8)

            cv2.namedWindow('ccc', 0)
            cv2.imshow('ccc', mask)
            cv2.waitKey()
        return
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue==0] = 0
        cv2.namedWindow('ccc', 0)
        cv2.imshow('ccc', labeled_img)
        cv2.waitKey()

            
    def pathsFromCC(self, num_labels, labels_img, skel_img):
        skel = skel_img.copy()

        endpoints_all = self.skeletonEndpointsList(skel)
        paths = {}
        for n in range(1, num_labels):
            endpoints_cc = [e for e in endpoints_all if labels_img[tuple([e[1], e[0]])] == n]
            if len(endpoints_cc) == 2: # should be always the case!
                path = self.walkFaster(skel, endpoints_cc[0])
                if len(path) > 0:
                    paths[n] = path  
                #else:
                #    print("short path for label ", n) 
        return paths


    def walkFaster(self, skel, start):
        
        path = [(int(start[1]), int(start[0]))]
        end = False
        while not end:
            end = True
            act = path[-1]
            skel[act[0], act[1]] = 0.
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                if skel[act[0] + dx, act[1] + dy]:
                    aim_x = act[0] + dx
                    aim_y = act[1] + dy
                    path.append((aim_x, aim_y))
                    end = False
                    break

        path = np.array(path)
        path -= 1

        if self.drop_endpoints:
            path = path[1:-1]

        return path


    def skeletonEndpoints(self, skel):
        skel = skel.copy()
        skel[skel!=0] = 1
        skel = np.uint8(skel)

        # Apply the convolution.
        kernel = np.uint8([[1,  1, 1],
                        [1, 10, 1],
                        [1,  1, 1]])
        src_depth = -1
        filtered = cv2.filter2D(skel,src_depth,kernel)

        p = np.where(filtered==11)
        return np.array([p[0], p[1]])


    def skeletonEndpointsList(self, skel):
        endpoints = self.skeletonEndpoints(skel)
       
        for e in endpoints:
            if e.shape[0] == 0:
                return []
        
        return list(zip(endpoints[1], endpoints[0]))


    def skeletonIntpoints(self, skel):
        skel = skel.copy()
        skel[skel!=0] = 1
        skel = np.uint8(skel)

        # Apply the convolution.
        kernel = np.uint8([[1,  1, 1],
                        [1, 10, 1],
                        [1,  1, 1]])
        src_depth = -1
        filtered = cv2.filter2D(skel,src_depth,kernel)

        p = np.where(filtered>12)
        return np.array([p[0], p[1]])
