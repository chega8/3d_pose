import os
import cv2
import time
import torch
import shutil
import numpy as np
import os.path as osp
from tqdm import tqdm
from torch.utils.data import DataLoader
print(__file__)
import imp

torchvision = imp.load_source('torchvision', '/usr/local/lib/python3.6/dist-packages/torchvision/__init__.py')
mmdet = imp.load_source('mmdet', '/content/gdrive/My Drive/vibe_repos/mmdetection/mmdet/__init__.py')


from torchvision.models.detection import keypointrcnn_resnet50_fpn
from yolov3.yolo import YOLOv3

from multi_person_tracker import Sort
from multi_person_tracker.data import ImageFolder, images_to_video

from google.colab.patches import cv2_imshow

from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.parallel import collate, scatter

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


class MPT():
    def __init__(
            self,
            device=None,
            batch_size=12,
            display=False,
            detection_threshold=0.7,
            detector_type='yolo',
            yolo_img_size=608,
            output_format='list',
            detector_checkpoint=None,
            detector_config=None
    ):
        '''
        Multi Person Tracker

        :param device (str, 'cuda' or 'cpu'): torch device for model and inputs
        :param batch_size (int): batch size for detection model
        :param display (bool): display the results of multi person tracking
        :param detection_threshold (float): threshold to filter detector predictions
        :param detector_type (str, 'maskrcnn' or 'yolo'): detector architecture
        :param yolo_img_size (int): yolo detector input image size
        :param output_format (str, 'dict' or 'list'): result output format
        '''

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.batch_size = batch_size
        self.display = display
        self.detection_threshold = detection_threshold
        self.output_format = output_format
        self.detector_type = detector_type
        self.detector_checkpoint = detector_checkpoint[0] if type(detector_checkpoint) == tuple else detector_checkpoint,
        self.detector_config = detector_config[0] if type(detector_config) == tuple else detector_config
    

        if self.detector_type == 'maskrcnn':
            self.detector = keypointrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        elif self.detector_type == 'yolo':
            self.detector = YOLOv3(
                device=self.device, img_size=yolo_img_size, person_detector=True, video=True, return_dict=True
            )
            # output [{'boxes': tensor([], size=(0, 4)), 
            #           'scores': tensor([]), 
            #           'classes': tensor([])}]
            # x = torch.Tensor([np.random.rand(3, 300, 400), np.random.rand(3, 300, 400)])
            # print(self.detector(x))
        elif self.detector_type == 'retina':
            self.detector = init_detector(
                self.detector_config, self.detector_checkpoint[0], device='cuda:0'
            )
        else:
            raise ModuleNotFoundError

        self.tracker = Sort()

    @torch.no_grad()
    def run_tracker(self, dataloader):
        '''
        Run tracker on an input video

        :param video (ndarray): input video tensor of shape NxHxWxC. Preferable use skvideo to read videos
        :return: trackers (ndarray): output tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        '''

        # initialize tracker
        self.tracker = Sort()

        start = time.time()
        print('Running Multi-Person-Tracker')
        trackers = []
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)

            # TODO: add handler for own detector input batch format
            if self.detector_type == 'retina':
                predictions = batch_inferense(self.detector, batch)
            else:
                predictions = self.detector(batch)
            
            for pred in predictions:
                print('type(pred)', type(pred))
                bb = pred['boxes'].cpu().numpy()
                sc = pred['scores'].cpu().numpy()[..., None]
                dets = np.hstack([bb,sc])
                dets = dets[sc[:,0] > self.detection_threshold]

                # if nothing detected do not update the tracker
                if dets.shape[0] > 0:
                    track_bbs_ids = self.tracker.update(dets)
                else:
                    track_bbs_ids = np.empty((0, 5))
                trackers.append(track_bbs_ids)

        runtime = time.time() - start
        fps = len(dataloader.dataset) / runtime
        print(f'Finished. Detection + Tracking FPS {fps:.2f}')
        return trackers

    def prepare_output_tracks(self, trackers):
        '''
        Put results into a dictionary consists of detected people
        :param trackers (ndarray): input tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        :return: dict: of people. each key represent single person with detected bboxes and frame_ids
        '''
        people = dict()

        for frame_idx, tracks in enumerate(trackers):
            for d in tracks:
                person_id = int(d[4])
                # bbox = np.array([d[0], d[1], d[2] - d[0], d[3] - d[1]]) # x1, y1, w, h

                w, h = d[2] - d[0], d[3] - d[1]
                c_x, c_y = d[0] + w/2, d[1] + h/2
                w = h = np.where(w / h > 1, w, h)
                bbox = np.array([c_x, c_y, w, h])

                if person_id in people.keys():
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
                else:
                    people[person_id] = {
                        'bbox' : [],
                        'frames' : [],
                    }
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
        for k in people.keys():
            people[k]['bbox'] = np.array(people[k]['bbox']).reshape((len(people[k]['bbox']), 4))
            people[k]['frames'] = np.array(people[k]['frames'])

        return people

    def display_results(self, image_folder, trackers, output_file=None):
        '''
        Display the output of multi-person-tracking
        :param video (ndarray): input video tensor of shape NxHxWxC
        :param trackers (ndarray): tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        :return: None
        '''
        print('Displaying results..')

        save = True if output_file else False
        tmp_write_folder = osp.join('/tmp', f'{osp.basename(image_folder)}_mpt_results')
        os.makedirs(tmp_write_folder, exist_ok=True)

        colours = np.random.rand(32, 3)
        image_file_names = sorted([
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for idx, (img_fname, tracker) in enumerate(zip(image_file_names, trackers)):

            img = cv2.imread(img_fname)
            for d in tracker:
                d = d.astype(np.int32)
                c = (colours[d[4] % 32, :] * 255).astype(np.uint8).tolist()
                cv2.rectangle(
                    img, (d[0], d[1]), (d[2], d[3]),
                    color=c, thickness=int(round(img.shape[0] / 256))
                )
                cv2.putText(img, f'{d[4]}', (d[0] - 9, d[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(img, f'{d[4]}', (d[0] - 8, d[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            # cv2.imshow('result video', img)
            cv2_imshow(img)

            # time.sleep(0.03)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if save:
                cv2.imwrite(osp.join(tmp_write_folder, f'{idx:06d}.png'), img)

        cv2.destroyAllWindows()

        if save:
            print(f'Saving output video to {output_file}')
            images_to_video(img_folder=tmp_write_folder, output_vid_file=output_file)
            shutil.rmtree(tmp_write_folder)


    def __call__(self, image_folder, output_file=None):
        '''
        Execute MPT and return results as a dictionary of person instances

        :param video (ndarray): input video tensor of shape NxHxWxC
        :return: a dictionary of person instances
        '''

        image_dataset = ImageFolder(image_folder)
        print('IMAGE_FOLDER_____________', image_folder)
        dataloader = DataLoader(image_dataset, batch_size=self.batch_size, num_workers=8)

        trackers = self.run_tracker(dataloader)
        if self.display:
            self.display_results(image_folder, trackers, output_file)

        if self.output_format == 'dict':
            result = self.prepare_output_tracks(trackers)
        elif self.output_format == 'list':
            result = trackers

        return result


def prepare_image(model, img):
    # class LoadImagee(object):

    #     def __call__(self, results):
    #         if isinstance(results['img'], str):
    #             results['filename'] = ''#results['img']
    #         else:
    #             results['filename'] = ''
    #         # img = mmcv.imread(results['img'])
    #         # img = np.random.randint(0, 255, (720, 1280, 3))
    #         results['img'] = np.float32(results['img'].cpu().numpy())
    #         img = results['img']
    #         results['img_shape'] = img.shape
    #         results['ori_shape'] = img.shape
    #         return results

    class LoadImage(object):

        def __call__(self, results):
            if isinstance(results['img'], str):
                results['filename'] = results['img']
            else:
                results['filename'] = None
            img = mmcv.imread(results['img'])
            results['img'] = img
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            return results

    # img = '/content/gdrive/My Drive/catapulta/Overhead_train_images/frame10000.jpg'
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImagee()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    return data

def predict(model, data):
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def match_output(mmdet_out):
    out = {
        'boxes': [], 
        'scores': [], 
        'classes': []
        }
    for box in mmdet_out[0]:
        if box[4] > 0.3:
            out['boxes'].append([box[0], box[1], box[2], box[3]])
            out['scores'].append(box[4])
            out['classes'].append(0)
    # for k in out.keys():
    #     out[k] = torch.Tensor(out[k])
    print('out', out)
    return out

def batch_inferense(model, batch):
    result = []
    for b in batch:
        data = prepare_image(model, b)
        result = predict(model, data)
        result.append(match_output(result))
    print(result)
    return result