import csv
import os

class DataLoaderDB:
    def __init__(self, path, dataloader=None):
        self.path = path
        self.dataloader = dataloader

    def get_value_by_id(self, image_id, class_id):
        # 讀取 CSV 檔案，尋找符合 image_id 和 class_id 的資料列
        get = []
        with open(self.path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['image_id'] == str(image_id) and row['class_id'] == str(class_id):
                    get.append(row)
        return get
        # return None

    def get_image_ids(self):
        """获取所有唯一的 image_id"""
        image_ids = set()
        with open(self.path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_ids.add(row['image_id'])
        return list(image_ids)
    
    def get_class_ids_by_image_id(self, image_id):
        """根据 image_id 获取所有关联的 class_id"""
        class_ids = {}

        with open(self.path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['image_id'] == str(image_id):
                    class_ids[row['class_id']] = class_ids.get(row['class_id'], 0) + 1
        return class_ids
    
    def get_specific_data( self , image_id, class_id, key):
        """获取指定 image_id 和 class_id 的特定键值"""
        get = []
        img = self.dataloader._get_dataset_image_by_id(image_id)
        w, h = img.size
        with open(self.path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['image_id'] == str(image_id) and row['class_id'] == str(class_id):
                    if key == "point1_x" or  key == "point2_x" or key == "detection_point1_x" or key == "detection_point2_x" or key == "context_roi_point1_x" or key == "context_roi_point2_x":
                        row[key] = float(row[key]) * w
                    elif key == "point1_y" or  key == "point2_y" or key == "detection_point1_y" or key == "detection_point2_y" or key == "context_roi_point1_y" or key == "context_roi_point2_y":
                        row[key] = float(row[key]) * h
                    get.append(row[key])

        return get

    def get_ioU_list_by_ids(self, image_id, class_id):
        # 获取真实框坐标
        try:
            truth_box_point1_x = self.get_specific_data(image_id, class_id, "point1_x")
            truth_box_point1_y = self.get_specific_data(image_id, class_id, "point1_y")
            truth_box_point2_x = self.get_specific_data(image_id, class_id, "point2_x")
            truth_box_point2_y = self.get_specific_data(image_id, class_id, "point2_y")
            
            # 获取检测框坐标
            detection_point1_x = self.get_specific_data(image_id, class_id, "detection_point1_x")
            detection_point1_y = self.get_specific_data(image_id, class_id, "detection_point1_y")
            detection_point2_x = self.get_specific_data(image_id, class_id, "detection_point2_x")
            detection_point2_y = self.get_specific_data(image_id, class_id, "detection_point2_y")

            if detection_point1_x is None or len(detection_point1_x) == 0 or detection_point1_x == "":
                print(f"No detection data found for image_id={image_id}, class_id={class_id}")
                return []
            # 构建真实框和检测框列表X
            truth_boxes = []
            detection_boxes = []
            for i in range(len(truth_box_point1_x)):
                # 真实框 (左上角, 右下角)
                truth_box = (
                    (float(truth_box_point1_x[i]), float(truth_box_point1_y[i])),
                    (float(truth_box_point2_x[i]), float(truth_box_point2_y[i]))
                )
                truth_boxes.append(truth_box)
                
                # 检测框 (左上角, 右下角)
                detection_box = (
                    (float(detection_point1_x[i]), float(detection_point1_y[i])),
                    (float(detection_point2_x[i]), float(detection_point2_y[i]))
                )
                detection_boxes.append(detection_box)
            
            # 计算每个框对的 IoU
            iou_list = []
            for truth_box, detect_box in zip(truth_boxes, detection_boxes):
                # 提取坐标值
                truth_x1, truth_y1 = truth_box[0]
                truth_x2, truth_y2 = truth_box[1]
                detect_x1, detect_y1 = detect_box[0]
                detect_x2, detect_y2 = detect_box[1]
                
                # 计算交集区域
                inter_x1 = max(truth_x1, detect_x1)
                inter_y1 = max(truth_y1, detect_y1)
                inter_x2 = min(truth_x2, detect_x2)
                inter_y2 = min(truth_y2, detect_y2)
                
                # 计算交集面积 (确保没有负值)
                inter_width = max(0, inter_x2 - inter_x1)
                inter_height = max(0, inter_y2 - inter_y1)
                intersection_area = inter_width * inter_height
                
                # 计算各自面积
                truth_area = (truth_x2 - truth_x1) * (truth_y2 - truth_y1)
                detect_area = (detect_x2 - detect_x1) * (detect_y2 - detect_y1)
                
                # 计算并集面积
                union_area = truth_area + detect_area - intersection_area
                
                # 计算 IoU (避免除以零)
                iou = intersection_area / union_area if union_area > 0 else 0.0
                iou_list.append(iou)
            
            return iou_list
        except Exception as e:
            print(f"Error while calculating IoU for image_id={image_id}, class_id={class_id}: {e}")
            return []

    def _get_box_distance(self, box1 , box2 ):
        """
        计算两个边界框之间的距离
        box1 和 box2 格式: ((x1,y1), (x2,y2))
        """
        # 提取 box1 坐标
        box1_x1, box1_y1 = box1[0]
        box1_x2, box1_y2 = box1[1]
        
        # 提取 box2 坐标
        box2_x1, box2_y1 = box2[0]
        box2_x2, box2_y2 = box2[1]
        
        # 计算中心点坐标
        center_box1_x = (box1_x1 + box1_x2) / 2
        center_box1_y = (box1_y1 + box1_y2) / 2
        center_box2_x = (box2_x1 + box2_x2) / 2
        center_box2_y = (box2_y1 + box2_y2) / 2
        
        # 计算欧氏距离
        distance = ((center_box1_x - center_box2_x) ** 2 + (center_box1_y - center_box2_y) ** 2) ** 0.5
        return distance

    def write_detect_point_to_db_by_ids(self, image_id, class_id, values):
        # 從資料庫獲取所有 truth box 資料
        truth_box_point1_x = self.get_specific_data(image_id, class_id, "point1_x")
        truth_box_point1_y = self.get_specific_data(image_id, class_id, "point1_y")
        truth_box_point2_x = self.get_specific_data(image_id, class_id, "point2_x")
        truth_box_point2_y = self.get_specific_data(image_id, class_id, "point2_y")
        img = self.dataloader._get_dataset_image_by_id(image_id)
        width, height = img.size
        # 構建 truth box 列表
        truth_boxes = []
        for i in range(len(truth_box_point1_x)):
            truth_box = (
                (float(truth_box_point1_x[i] / width ), float(truth_box_point1_y[i] / height)),
                (float(truth_box_point2_x[i] / width), float(truth_box_point2_y[i] / height))
            )
            truth_boxes.append(truth_box)
        
        # 為每個 truth box 尋找匹配的 detection box
        matched_pairs = []
        
        for truth_idx, truth_box in enumerate(truth_boxes):
            best_match = None
            best_iou = -1
            best_distance = float('inf')
            
            # 第一步：尋找 IoU 最大的 detection box
            for detect_idx, detect_box in enumerate(values):
                iou = self.compute_iou_for_pair(truth_box, detect_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = detect_box
            
            # 第二步：如果 IoU=0，則使用距離匹配
            if best_iou <= 0:
                for detect_idx, detect_box in enumerate(values):
                    distance = self._get_box_distance(truth_box, detect_box)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = detect_box
            
            # 確保找到匹配
            if best_match is not None:
                matched_pairs.append((truth_box, best_match))
        
        # 將匹配結果寫入資料庫
        for truth_box, detect_box in matched_pairs:
            point1 = truth_box[0]
            point2 = truth_box[1]
            detect_point1 = detect_box[0]
            detect_point2 = detect_box[1]
            
            self.write_to_db_by_ids_and_truth_box(
                image_id=image_id,
                class_id=class_id,
                point1=point1,
                point2=point2,
                detect_point1=detect_point1,
                detect_point2=detect_point2
            )
        
        return len(matched_pairs)

    def compute_iou_for_pair(self, box1, box2):
        """
        计算两个边界框之间的 IoU
        box1 和 box2 格式: ((x1,y1), (x2,y2))
        """
        # 提取 box1 坐标
        box1_x1, box1_y1 = box1[0]
        box1_x2, box1_y2 = box1[1]
        
        # 提取 box2 坐标
        box2_x1, box2_y1 = box2[0]
        box2_x2, box2_y2 = box2[1]
        
        # 计算交集区域
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        # 计算交集面积
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        intersection_area = inter_width * inter_height
        
        # 计算各自面积
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        
        # 计算并集面积
        union_area = box1_area + box2_area - intersection_area
        
        # 计算 IoU (避免除以零)
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou
    def normalize_point(self, point, width, height):
        """
        將點的座標歸一化到 [0, 1] 範圍
        point: (x, y)
        width: 圖像寬度
        height: 圖像高度
        """
        if point[0] > 1 or point[1] > 1:
            return (point[0] / width, point[1] / height)
        return point

    def normalize_points(self, points, image_id):
        """
        將一組點的座標歸一化到 [0, 1] 範圍
        points: [(x1, y1), (x2, y2), ...]
        width: 圖像寬度
        height: 圖像高度
        """
        img = self.dataloader._get_dataset_image_by_id(image_id)
        width, height = img.size
        ret = []
        for point in points:
            if point == None:
                ret.append(None)
            else:
                ret.append(self.normalize_point(point, width, height))
        return ret

    def write_to_db_by_ids_and_truth_box(self, image_id, class_id, point1, point2, detect_point1, detect_point2):
        """
        以 image_id, class_id, point1, point2 為複合 key 查找，並寫入 detection_point1_x, detection_point1_y, detection_point2_x, detection_point2_y
        detect_point1, detect_point2 格式為 (x, y)
        """
        # 讀取所有資料
        rows = []
        updated = False
        points = [point1, point2, detect_point1, detect_point2]
        point1, point2, detect_point1, detect_point2 = \
            self.normalize_points( points , image_id )
        with open(self.path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            for row in reader:
                if (row['image_id'] == str(image_id) and 
                    row['class_id'] == str(class_id) and
                    row['point1_x'] == str(point1[0]) and
                    row['point1_y'] == str(point1[1]) and
                    row['point2_x'] == str(point2[0]) and
                    row['point2_y'] == str(point2[1])):
                    # 寫入 detection 欄位
                    row['detection_point1_x'] = detect_point1[0]
                    row['detection_point1_y'] = detect_point1[1]
                    row['detection_point2_x'] = detect_point2[0]
                    row['detection_point2_y'] = detect_point2[1]
                    updated = True
                rows.append(row)

        # 有更新才寫回
        if updated:
            with open(self.path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            print(f"No matching row for image_id={image_id}, class_id={class_id}, point1={point1}, point2={point2}")
        return updated

    
    def initialize_csv(self):
        """確保 CSV 檔案存在且有標題列"""
        # Check if file exists and delete it
        if os.path.exists(self.path):
            os.remove(self.path)
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "image_id", "class_id", "point1_x", "point1_y", "point2_x", "point2_y",
                "detection_point1_x", "detection_point1_y", "detection_point2_x", "detection_point2_y",
                "context_roi_point1_x", "context_roi_point1_y", "context_roi_point2_x", "context_roi_point2_y"
            ])
    def if_exists(self, image_id, class_id, point1, point2 ):
        """檢查是否已存在相同的 image_id 和 class_id"""
        with open(self.path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if (row['image_id'] == str(image_id) and 
                    row['class_id'] == str(class_id) and
                    row['point1_x'] == str(point1[0]) and
                    row['point1_y'] == str(point1[1]) and
                    row['point2_x'] == str(point2[0]) and
                    row['point2_y'] == str(point2[1])):
                    return True
        return False
    def write_context_roi_to_db(self, image_id, class_id, point1, point2, context_roi_point1, context_roi_point2):
        """
        更新已存在的记录，添加或更新 context ROI 信息
        
        Args:
            image_id: 图像ID
            class_id: 类别ID
            point1: 边界框左上角坐标 (x, y)
            point2: 边界框右下角坐标 (x, y)
            context_roi_point1: context ROI 左上角坐标 (x, y)
            context_roi_point2: context ROI 右下角坐标 (x, y)
        """
        # 检查记录是否存在
        points = [point1, point2, context_roi_point1, context_roi_point2]
        point1, point2, context_roi_point1, context_roi_point2 = \
            self.normalize_points( points , image_id )
        if not self.if_exists(image_id, class_id, point1, point2):
            print(f"Data for image_id {image_id} and class_id {class_id} does not exist. Cannot write context ROI.")
            return False
        # 读取所有行
        rows = []
        fieldnames = None
        updated = False
        
        with open(self.path, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            fieldnames = next(reader)  # 读取标题行
            for row in reader:
                # 检查是否匹配目标记录
                if (row[0] == str(image_id) and 
                    row[1] == str(class_id) and
                    row[2] == str(point1[0]) and
                    row[3] == str(point1[1]) and
                    row[4] == str(point2[0]) and
                    row[5] == str(point2[1])):
                    
                    # 更新 context ROI 字段
                    row[10] = str(context_roi_point1[0])  # context_roi_point1_x
                    row[11] = str(context_roi_point1[1])  # context_roi_point1_y
                    row[12] = str(context_roi_point2[0])  # context_roi_point2_x
                    row[13] = str(context_roi_point2[1])  # context_roi_point2_y
                    updated = True
                rows.append(row)
        
        # 如果有更新，写回文件
        if updated:
            with open(self.path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)  # 写入标题行
                writer.writerows(rows)  # 写入所有行
            return True
        else:
            print(f"Found matching record but failed to update context ROI for image_id={image_id}, class_id={class_id}")
            return False

    def write_to_db(self, image_id, class_id, point1=None, point2=None,
                   detect_point1=None, detect_point2=None,
                   context_roi_point1=None, context_roi_point2=None):
        """寫入一筆資料到 CSV 檔案"""
        if self.if_exists(image_id, class_id, point1, point2):
            print(f"Data for image_id {image_id} and class_id {class_id} already exists.")
            return
        points = [point1, point2, detect_point1, detect_point2, context_roi_point1, context_roi_point2]
        point1, point2, detect_point1, detect_point2, context_roi_point1, context_roi_point2 = \
            self.normalize_points( points , image_id )

        with open(self.path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                image_id, class_id, 
                point1[0] if point1 else "", 
                point1[1] if point1 else "",
                point2[0] if point2 else "",
                point2[1] if point2 else "",
                detect_point1[0] if detect_point1 else "",
                detect_point1[1] if detect_point1 else "",
                detect_point2[0] if detect_point2 else "",
                detect_point2[1] if detect_point2 else "",
                context_roi_point1[0] if context_roi_point1 else "",
                context_roi_point1[1] if context_roi_point1 else "",
                context_roi_point2[0] if context_roi_point2 else "",
                context_roi_point2[1] if context_roi_point2 else ""
            ])

