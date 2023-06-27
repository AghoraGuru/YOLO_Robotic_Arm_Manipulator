from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import rclpy

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.bridge = cv2.CvBridge()
        self.image_plotted = False

    def callback(self, msg):
        # Convert the ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        if not self.image_plotted:
            # Process the image using your YOLO object detection code
            # Pass cv_image to your YOLO object detection code
            model = YOLO(model='best.pt')
            results = model(cv_image)
            
            boxes = results[0].cpu().boxes
            
            res_plotted = results[0].plot()
            
            x = []
            y = []
            for box in boxes:
                box = box.xyxy

                if (box[0][2] - box[0][0]) > 70:
                    y.append((box[0][1] + box[0][3]) / 2)
                    x.append((box[0][2]))

                else:
                    x.append((box[0][0] + box[0][2]) / 2)
                    y.append((box[0][1] + box[0][3]) / 2)

            plt.scatter(x, y, c='r', s=40)
            for a, b, in zip(x, y):
                x[x.index(a)] = (a.numpy() - 500) / 1000
                y[y.index(b)] = (500 - b.numpy()) / 1000 + 0.45


            objects = {}
            iter = 0
            for i in boxes.cls.numpy():
                if i not in objects.keys():
                    objects[results[0].names[i]] = []
                    objects[results[0].names[i]].append((x[iter], y[iter]))
                else:
                    objects[results[0].names[i]].append((x[iter].numpy(), y[iter].numpy()))

                iter += 1

            # Plot the received image
            plt.imshow(cv_image)
            plt.axis('off')
            plt.show()
            self.image_plotted = True