import os
import logging
import boto3
import io
import json
import random
import string

from mmdet.apis import init_detector, inference_detector
import tke
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse
import anyconfig
import cv2
import torch
from torchvision import transforms
logger = logging.getLogger(__name__)
def get_transforms( ):
    transforms_config=[
{'type':'ToTensor','args':{}},
{'type':'Normalize','args':{
    'mean':[0.485, 0.456, 0.406],
    'std':[0.229, 0.224, 0.225]
}}]
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']
        cls = getattr(transforms, item['type'])(**args)
        tr_list.append(cls)
    tr_list = transforms.Compose(tr_list)
    return tr_list

class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(self, config_file=None,
                 checkpoint_file=None,
                 image_dir=None,
                 labels_file=None, score_threshold=0.3, device='cpu', **kwargs):
        """
        Load MMDetection model from config and checkpoint into memory.
        (Check https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html#high-level-apis-for-testing-images)

        Optionally set mappings from COCO classes to target labels
        :param config_file: Absolute path to MMDetection config file (e.g. /home/user/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x.py)
        :param checkpoint_file: Absolute path MMDetection checkpoint file (e.g. /home/user/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs: can contain endpoint_url in case of non amazon s3
        """
        super(MMDetection, self).__init__(**kwargs)
        config_file = config_file or os.environ['config_file']
        checkpoint_file = checkpoint_file or os.environ['checkpoint_file']
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file
        self.endpoint_url = kwargs.get('endpoint_url')
        if self.endpoint_url:
            logger.info(f'Using s3 endpoint url {self.endpoint_url}')
        self.device = device
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}
        # self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
        #     self.parsed_label_config, 'PolygonLabels', 'Image')
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(  # noqa E501
            {'PolygonLabels': self.parsed_label_config['PolygonLabels']}, 'PolygonLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name
        config = anyconfig.load(open(config_file, 'rb'))
        print('Load new model from: ', config_file, checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        self.model = tke.build_model(config['arch']).to(device)
        self.model.load_state_dict(checkpoint['state_dict'], strict =False)
        self.score_thresh = score_threshold
        self.transformer = get_transforms()

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3', endpoint_url=self.endpoint_url)
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url
    def pre_process(self, image_path,short_size=736):
        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, _ = im.shape
        short_edge = h
        scale = short_size / short_edge
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        im = self.transformer(im)
        return im,(h,w)
    def predict(self, tasks, **kwargs):
        
        assert len(tasks) != 0
        results = []
        for task in tasks:
        # task = tasks[0]
        
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            all_scores = []
            image,shape = self.pre_process(image_path)
            image = torch.unsqueeze(image, 0,).to(device=self.device)
            outputs = self.model(image, {'shape':[shape]},is_training = False)['coarse_polys']
            # points_list = []
            for output in outputs[0]:
                points = []
                for point in output:
                    x, y = point[0], point[1]
                    points.append([float(x)/shape[1]*100,
                                    float(y)/shape[0] * 100])
                print(points)
                results.append({
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "original_width": shape[1],
                    "original_height": shape[0],
                    # "image_rotation": 0,
                    "value": {
                        "points": points,
                        "polygonlabels": ['text'],
                    },
                    "type": "polygonlabels",
                    "id": ''.join(random.SystemRandom().choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6)), # creates a random ID for your label every time
                    "readonly": False,
                })
        return [{'result': results}]
        # for bboxes, label in zip(model_results, self.model.CLASSES):
            # output_label = self.label_map.get(label, label)

            # if output_label not in self.labels_in_config:
            #     print(output_label + ' label not found in project config.')
            #     continue
        # for bbox,label,score in zip(model_results.pred_instances.bboxes,model_results.pred_instances.labels,model_results.pred_instances.scores) :
        #     # print(bboxes)
        #     # for bbox in bboxes:
        #     #     bbox = list(bbox)
        #     #     if not bbox:
        #     #         continue
        #     #     score = float(bbox[-1])
        #     #     if score < self.score_thresh:
        #     #         continue
        #     bbox = [float(i) for i in bbox]
        #     score = float(score.cpu())
        #     x, y, xmax, ymax = bbox[:4]
        #     results.append({
        #         'from_name': self.from_name,
        #         'to_name': self.to_name,
        #         'type': 'rectanglelabels',
        #         'value': {
        #             # 'rectanglelabels': [output_label],
        #             'rectanglelabels': ['Airplane'],
        #             'x': x / img_width * 100,
        #             'y': y / img_height * 100,
        #             'width': (xmax - x) / img_width * 100,
        #             'height': (ymax - y) / img_height * 100
        #         },
        #         'score': score
        #     })
        #     all_scores.append(score)
        # avg_score = sum(all_scores) / max(len(all_scores), 1)
        # return {
        #     'result': results,
        #     'score': avg_score
        # }


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
