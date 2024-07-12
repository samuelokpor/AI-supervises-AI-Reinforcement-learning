import gymnasium as gym
from gymnasium import spaces
import os
import numpy as np
import cv2
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.env_checker import check_env
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ImageBBOx import ImageBBoxLabelList
import imgaug.augmenters as iaa
from collections import Counter
from torchvision.transforms import functional as F
import torchvision.transforms as transforms

#Load the dataset and Augment
# Image Augmentation
iaa_aug = iaa.Sequential([
    iaa.GaussianBlur((0, 1.0)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02 * 255), per_channel=False),
    iaa.Multiply((0.8, 1.25), per_channel=False),
    iaa.Add((0, 30), per_channel=False),
    iaa.LinearContrast((0.8, 1.25), per_channel=False),
    iaa.Affine(scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
    iaa.PerspectiveTransform(scale=(0, 0.01)),
], random_order=False)

# Load the dataset
ibll = ImageBBoxLabelList.from_pascal_voc('data/train_NG') 

# Define transformations
# Note: Adjust transformations as needed based on your specific requirements
train_tfms = [iaa.Resize({'width': 640, 'height': 480}), iaa_aug]
ibll.set_tfms(train_tfms)
ibll.apply_tfms()

# Split the dataset into training and testing (if not already split)
train_ibll, test_ibll = ibll.split(train_ratio=0.8)


label_counter = Counter()
for item in ibll.data:
    label_counter.update(item['labels'])

unique_labels = list(label_counter.keys())
label_map = {label: idx+1 for idx, label in enumerate(unique_labels)}  # +1 because 0 is usually reserved for background
print(unique_labels)
print(label_map)

MAX_DETECTIONS = 10  # Example maximum number of detections per image


def load_model(num_classes, model_path):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Current device before moving model: {device}")
    model = model.to(device)
    model.eval()
    return model

def check_and_fix_bbox(bbox):
    """
    Ensures that the bbox has a positive width and height.
    If not, it adjusts the bbox to have a small positive width and height.
    """
    xmin, ymin, xmax, ymax = bbox
    if xmax <= xmin:
        xmax = xmin + 0.1  # Ensure positive width
    if ymax <= ymin:
        ymax = ymin + 0.1  # Ensure positive height
    return [xmin, ymin, xmax, ymax]


def preprocess_image(image):
    """Preprocess the image for Faster R-CNN."""
    # Assuming 'image' is a NumPy array of shape (H, W, C) with pixel values in [0, 255]
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL image if necessary
        transforms.ToTensor(),  # Convert to PyTorch tensor and scale pixels to [0, 1]
    ])
    image = transform(image)  # Apply the transformations
    return image


def collate_fn(batch):
    return tuple(zip(*batch))

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, ibll, label_map):
        self.data = ibll.data
        self.label_map = label_map

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['img']
        bboxes = item['bboxes']
        labels = item['labels']

        # Convert image to tensor
        image = F.to_tensor(image)

        # Process bboxes and labels
        # Convert bboxes to a tensor. The model might expect a specific format, e.g., (x_min, y_min, x_max, y_max).
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        
        # Map labels to their corresponding integer IDs
        labels = [self.label_map[label] for label in labels]
        labels = torch.tensor(labels, dtype=torch.int64)
        
        # Create target dictionary expected by models like Faster R-CNN
        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        
        return image, target

class DefectDetectionRLEnv(gym.Env):
    def __init__(self, ibll, model, device):
        super(DefectDetectionRLEnv, self).__init__()
        self.dataset = ibll 
        self.model = model
        self.device = device
        self.current_image_index = 0
        self.label_map_inv = label_map_inv
        self.label_map = label_map
        self.last_predictions = None

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Example actions: 0=accept, 1=reject, 2=request additional analysis
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, 6), dtype=np.float32)

    def get_cnn_predictions(self, image, model, image_width, image_height, device, score_threshold=0.5):
        """Get predictions from the Faster R-CNN model."""
        # Ensure image is a PyTorch tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        # Move image to the device and add a batch dimension
      
        image = image.unsqueeze(0).to(self.device)
        
        
        # Perform inference
        with torch.no_grad():
            outputs = model(image)
        
        # Extract predictions for the first (and only) image in the batch
        output = outputs[0]
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()

        # Filter predictions with scores above a threshold
        high_confidence_idxs = scores > score_threshold
        boxes = boxes[high_confidence_idxs]
        labels = labels[high_confidence_idxs]
        scores = scores[high_confidence_idxs]

        # Normalize the bounding box coordinates
        # Assuming the image dimensions are known and represented by variables W and H
        # Example dimensions, adjust based on your actual image size
        boxes[:, [0, 2]] /= image_width # Normalize x coordinates by image width
        boxes[:, [1, 3]] /= image_height  # Normalize y coordinates by image height

        # Convert label IDs to class names if necessary
        class_names = [label_map_inv[label] for label in labels]  # Use the inverted label map from your script
        self.last_predictions = (boxes, labels, scores) #save the predictions
        #self.last_predictions = (np.array([[10, 10, 100, 100]]), ['Test'], [0.99])  # Example box coordinates and label
        return boxes, class_names, scores
    
    def encode_predictions(self, boxes, labels, scores):
        """
        Encode bounding box coordinates, labels, and scores into a fixed-size observation.
        
        Parameters:
        - boxes: Numpy array of bounding box coordinates with shape (num_detections, 4).
        - labels: List of labels for each detection.
        - scores: Numpy array of confidence scores for each detection.
        
        Returns:
        - Encoded observation as a torch tensor.
        """
        num_detections = len(scores)
        # Initialize observation array with zeros, considering the maximum number of detections
        observation = np.zeros((MAX_DETECTIONS, 6))  # 6 = 4 bbox coordinates + 1 label + 1 score
        
        
        # Fill in the observation with actual detections
        for i in range(min(num_detections, MAX_DETECTIONS)):
            observation[i, :4] = boxes[i]  # Bounding box coordinates
            label_id = self.label_map[labels[i]]
            observation[i, 4] = label_id  # Label
            observation[i, 5] = scores[i]  # Confidence score

        # Convert observation to a PyTorch tensor
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        
        return observation_tensor   
        
    # def step(self, action):
        
    #     # Retrieve the current image
    #     current_image = self.dataset[self.current_image_index][0]
    #     _, image_height, image_width = current_image.shape
        
    #     # Perform inference to get predictions
    #     boxes, labels, scores = self.get_cnn_predictions(current_image,
    #                                                      self.model,
    #                                                      image_width,
    #                                                      image_height=image_height,
    #                                                      device=self.device,
    #                                                      score_threshold=0.5)
        
    #     # Update last_predictions
    #     self.last_predictions = (boxes, labels, scores)
    #     #print(self.last_predictions)
    #     # Encode the predictions to form the next state
    #     next_state = self.encode_predictions(boxes, labels, scores)

    #     if isinstance(next_state, torch.Tensor):
    #         next_state = next_state.cpu().detach().numpy()
        
    #     # Initialize default values
    #     reward = 0
    #     info = {}
        
    #     # Define NG label index (ensure this matches your label_map)
    #     NG_LABEL_INDEX = self.label_map['NG']  # Assuming 'NG' is the key for defects in label_map
        
    #     # Check predictions and assign rewards based on specified logic
    #     if any(label == NG_LABEL_INDEX and score > 0.9 for label, score in zip(labels, scores)):
    #         # If the action is to accept the defect as truly present
    #         if action == 0:
    #             reward = 1  # Positive reward for correctly identifying a defect
    #         else:
    #             reward = -1  # Negative reward for missing the defect
    #     elif any(score < 0.5 for score in scores):
    #         # If the action is inconclusive due to low confidence
    #         if action == 2:  # Assuming action '2' corresponds to marking as inconclusive
    #             reward = -0.5  # Negative reward for inconclusiveness
    #         else:
    #             reward = -1  # Negative reward for wrong action when it should be inconclusive
        
    #     # Update the episode state
    #     self.current_image_index += 1
    #     terminated = self.current_image_index >= len(self.dataset)
    #     if terminated:
    #         self.current_image_index = 0  # Reset for the next episode
        
    #     truncated = False  # Assuming no truncation in this context
        
    #     # Return the necessary components
    #     return next_state, reward, terminated, truncated, info

    def step(self, action):
        current_image = self.dataset[self.current_image_index][0]
        _, image_height, image_width = current_image.shape
        
        boxes, labels, scores = self.get_cnn_predictions(current_image,
                                                        self.model,
                                                        image_width,
                                                        image_height,
                                                        self.device,
                                                        score_threshold=0.5)
        self.last_predictions = (boxes, labels, scores)
        next_state = self.encode_predictions(boxes, labels, scores)

        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().detach().numpy()
        
        reward = 0
        info = {}
        NG_LABEL_INDEX = self.label_map['NG']

        # Retrieve ground truth boxes for the current image
        ground_truths = self.dataset[self.current_image_index][1]['boxes']
        detected_ng = [(box, score) for box, label, score in zip(boxes, labels, scores) if label == NG_LABEL_INDEX]
        
        # Evaluate each detected box against the ground truth
        for box, score in detected_ng:
            iou_scores = [self.iou(box, gt) for gt in ground_truths]
            max_iou = max(iou_scores, default=0)
            if score > 0.9:
                if max_iou > 0.5:  # High confidence and good IoU
                    reward += 1 * max_iou  # Scale reward by IoU
                else:
                    reward -= 1  # Penalty for false positives
            else:
                reward -= 0.5  # Penalty for low confidence in high-IoU predictions

        # Handling the action taken by the agent
        if action == 0 and not detected_ng:
            reward -= 1  # Penalty for missing a defect when none were detected

        self.current_image_index += 1
        terminated = self.current_image_index >= len(self.dataset)
        if terminated:
            self.current_image_index = 0  # Reset for the next episode
        
        truncated = False
        return next_state, reward, terminated, truncated, info

    def reset(self, seed=None, return_info=False, options=None):
        # Optional: Implement seed handling if your environment uses random numbers
        if seed is not None:
            np.random.seed(seed)
            # If you use other libraries for random number generation, seed them here as well

        # Reset the environment to start from the first image or a random image if desired
        self.current_image_index = 0  # Or use np.random.randint(len(self.dataset)) for random start

        # Obtain the initial observation using the updated logic in _get_observation
        initial_observation = self._get_observation()

        info = {}  # Initialize an empty info dictionary

        # The reset method must always return a tuple (obs, info)
        return initial_observation, info

    def render(self, mode='human', action=None):
        if mode == 'human':
            image_data = self.dataset[self.current_image_index][0]
            print(image_data.shape)
            
            #convert from pytorch tp Numpy array amf change from [C, H, W] format
            if isinstance(image_data, torch.Tensor):
                #make sure the tensor is on CPU and detach it from the computaation graph
                image_data = image_data.cpu().detach()

                #convert to Numpy array
                image_data = image_data.numpy()
                #change from [C, H, W] to [H, W, C]
                image_data = np.transpose(image_data, (1, 2, 0))

            #Assuming the image data is now a numpy array in [H,W,C] format
            # Convert image data to uint8 if its not already
            if image_data.dtype != np.uint8:
                #Assuming your image data is normalized to [0,1], adjust accordingly
                image_data = (image_data * 255).astype(np.uint8)

            

            # Resize the image for display if necessary
            display_image = cv2.resize(image_data, (224, 224))

            # Draw the bounding boxes and labels on the resized display image
            if self.last_predictions is not None:
                boxes, labels, _ = self.last_predictions
                for box, label in zip(boxes, labels):
                    start_point = (int(box[0] * 224 / image_data.shape[1]), int(box[1] * 224 / image_data.shape[0]))  # Rescale box coords
                    end_point = (int(box[2] * 224 / image_data.shape[1]), int(box[3] * 224 / image_data.shape[0]))
                    cv2.rectangle(display_image, start_point, end_point, (255, 0, 0), 2)  # Draw rectangle
                    cv2.putText(display_image, str(label), start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Put label
            else:
                print("No Predictions available to display.")
            
            
            if action is not None:
                # Overlay the action on the image as text
                action_text = f"Action: {action}"
                cv2.putText(display_image, action_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
                
            cv2.imshow("Environment with Action", display_image)
            cv2.waitKey(0)  # Use cv2.waitKey(0) if you want the image window to wait until a key is pressed
        elif mode == 'rgb_array':
            # Return the current image as an RGB array
            return self.dataset[self.current_image_index][0]
        else:
            super(DefectDetectionRLEnv, self).render(mode=mode)  # Just in case there are other modes we can handle

    def _get_observation(self):
        # Retrieve the current image from the dataset
        current_image_tensor, annotations = self.dataset[self.current_image_index]
        #print(current_image_data)  # Add this to debug the structure
        
        # Preprocess the image for Faster R-CNN
        # Note: This step assumes the preprocessing (e.g., resize, normalization) aligns with the model's expectations
        preprocessed_image = preprocess_image(current_image_tensor)  # Ensure this function is defined as discussed

        _, image_height, image_width = preprocessed_image.shape
        # Perform inference with the Faster R-CNN model to get predictions
        boxes, labels, scores = self.get_cnn_predictions(preprocessed_image, self.model, image_width, image_height, self.device, score_threshold=0.5)  # Ensure this function is correctly implemented

        # Encode the predictions into a structured observation format
        # This step involves converting boxes, labels, and scores into the observation space format
        observation = self.encode_predictions(boxes, labels, scores)  # Ensure this function is defined to match your observation space

        return observation.cpu().numpy()
    
    
def visualize_episode(model, env, num_steps=100):
    obs, _ = env.reset()  # Adjusted to only receive the observation
    for step in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        env.render(action=action)  # Ensure this is correctly handled in your render method

        if done:
            print(f"Episode finished after {step+1} steps.")
            break

# Correctly prepare the dataset for the environment
# Example for dataset preparation (you'll need to adapt this to your specific setup):
images = [data['img'] for data in train_ibll.data]
targets = [{'bboxes': data['bboxes'], 'labels': data['labels']} for data in train_ibll.data]

# Initialize your custom dataset with the prepared data
custom_dataset = CustomDataset(train_ibll, label_map)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    label_map_inv = {v: k for k, v in label_map.items()}  
    print(label_map_inv)
    # Load the Faster R-CNN model for inference
    num_classes = len(label_map) + 1  # Including background
    faster_rcnn_model_path = "models/ct_testNG_7_model_60.pth"  # Update with your model path
    faster_rcnn_model = load_model(num_classes, faster_rcnn_model_path)  # Ensure this loads the pre-trained model
    print(f"Current device before moving model: {device}")
    faster_rcnn_model.to(device)

    faster_rcnn_model.eval()

    # Initialize the RL environment with the Faster R-CNN model for generating observations
    env = DefectDetectionRLEnv(custom_dataset, faster_rcnn_model, device)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    check_env(env, warn=True)

    # # Initialize the PPO model with Stable Baselines3 for RL training
    # ppo_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_defect_detection_tensorboard/", policy_kwargs=dict(normalize_images=False))
    
    # # Train the PPO model
    # total_timesteps = 100000  # Adjust based on your requirements
    # ppo_model.learn(total_timesteps=total_timesteps)
    
    # # Save the trained PPO model
    # ppo_model.save("ppo_defect_detection_model6")
    
    # print("Training complete! Model saved.")

     # Initialize the DQN model with Stable Baselines3 for RL training
    dqn_model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_defect_detection_tensorboard/", 
                    policy_kwargs=dict(net_arch=[256, 256]),  # Example architecture, adjust as needed
                    learning_rate=1e-4, 
                    buffer_size=50000, 
                    learning_starts=1000, 
                    batch_size=32, 
                    tau=1.0, 
                    gamma=0.99, 
                    train_freq=4, 
                    gradient_steps=1, 
                    target_update_interval=1000, 
                    exploration_fraction=0.1, 
                    exploration_initial_eps=1.0, 
                    exploration_final_eps=0.05)
    
    # Train the DQN model
    total_timesteps = 800000  # Adjust based on your requirements
    dqn_model.learn(total_timesteps=total_timesteps)
    
    # # Save the trained DQN model
    # dqn_model.save("dqn_defect_detection_model")

    # print("Training complete! Model saved.")


    # # Load the trained PPO model
    # ppo_model = PPO.load("ppo_defect_detection_model5.zip")

    # # Visualize the behavior of the loaded model
    # visualize_episode(ppo_model, env, num_steps=100)

    

    # #Load the trained DQN model
    # dqn_model = DQN.load("dqn_defect_detection_model.zip", env=env)

    # #Visualize the behaviour of the loaded model
    # visualize_episode(dqn_model, env, num_steps=100)



