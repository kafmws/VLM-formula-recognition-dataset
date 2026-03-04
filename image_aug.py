from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageTransform
import random
import numpy as np
import cv2
from io import BytesIO
import imgaug.augmenters as iaa  # 可选，用于复杂增强

class LatexAugmentation:
    """针对 LaTeX 公式识别的数据增强，每次随机选择2-3种"""
    
    def __init__(self, 
                 apply_prob: float = 0.5,  # 整体应用增强的概率
                 min_ops: int = 2,          # 最少增强操作数
                 max_ops: int = 3):         # 最多增强操作数
        
        self.apply_prob = apply_prob
        self.min_ops = min_ops
        self.max_ops = max_ops
        
        # 定义所有可用的增强操作（按类别分组）
        self.augmentations = {
            # 1. 几何与空间变换
            'geometry': [
                self.rotate,
                self.perspective,
                self.affine,
                self.lens_distortion,
                self.canvas_expand_trim,
            ],
            # 2. 颜色与光照调整
            'color': [
                self.color_jitter,
                self.rgb_shift,
                self.color_temperature,
                self.gamma_correction,
                self.channel_dropout,
            ],
            # 3. 噪声干扰
            'noise': [
                self.gaussian_noise,
                self.salt_pepper_noise,
                self.poisson_noise,
                self.speckle_noise,
            ],
            # 4. 模糊与画质损伤
            'blur': [
                self.gaussian_blur,
                self.motion_blur,
                self.jpeg_artifacts,
            ]
        }
        
        # 为每个类别分配权重（可以根据需要调整）
        self.category_weights = {
            'geometry': 0.3,
            'color': 0.3,
            'noise': 0.2,
            'blur': 0.2
        }
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """应用随机增强"""
        
        # 以一定概率不应用增强
        if random.random() > self.apply_prob:
            return image
        
        # 保存原始模式
        original_mode = image.mode
        if original_mode != 'RGB':
            image = image.convert('RGB')
        
        # 随机决定本次要应用的增强数量
        num_ops = random.randint(self.min_ops, self.max_ops)
        
        # 根据权重选择增强类别
        categories = list(self.category_weights.keys())
        weights = list(self.category_weights.values())
        weights = [w / sum(weights) for w in weights]
        
        selected_categories = []
        for _ in range(num_ops):
            if len(selected_categories) == len(categories):
                # 如果已经选了所有类别，随机重复
                category = random.choice(categories)
            else:
                # 从未选的类别中按权重选择
                available = [c for c in categories if c not in selected_categories]
                avail_weights = [self.category_weights[c] for c in available]
                avail_weights = [w / sum(avail_weights) for w in avail_weights]
                category = random.choices(available, weights=avail_weights)[0]
            
            selected_categories.append(category)
        
        # 为每个选中的类别随机选择一个具体的增强操作
        applied_ops = []
        for category in selected_categories:
            ops = self.augmentations[category]
            selected_op = random.choice(ops)
            applied_ops.append(selected_op.__name__)
            # print(selected_op.__name__)
            
            # 应用增强
            image = selected_op(image)
        
        # 恢复原始模式
        if original_mode != 'RGB' and original_mode != image.mode:
            image = image.convert(original_mode)
        
        return image
    
    # ========== 几何与空间变换 ==========
    
    def rotate(self, image: Image.Image) -> Image.Image:
        """旋转 (-5° 到 +5°)"""
        angle = random.uniform(-1, 1)
        return image.rotate(angle, expand=False, fillcolor=255, resample=Image.BICUBIC)
    
    def perspective(self, image: Image.Image) -> Image.Image:
        """透视变换（模拟拍摄角度变化）"""
        w, h = image.size
        
        # 随机偏移量
        shift_x = random.uniform(-0.02, 0.02) * w
        shift_y = random.uniform(-0.02, 0.02) * h
        
        # 定义变换前后的四个角点
        src_points = [(0, 0), (w, 0), (w, h), (0, h)]
        dst_points = [
            (shift_x, shift_y),
            (w - shift_x, shift_y),
            (w - shift_x, h - shift_y),
            (shift_x, h - shift_y)
        ]
        
        # 计算透视变换矩阵并应用
        coeffs = self.find_coeffs(src_points, dst_points)
        return image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    
    def affine(self, image: Image.Image) -> Image.Image:
        """仿射变换（缩放+错切）"""
        w, h = image.size
        
        # 随机缩放 (0.95-1.05)
        scale = random.uniform(0.95, 1.05)
        
        # 随机错切 (-0.02 到 0.02)
        shear_x = random.uniform(-0.02, 0.02)
        shear_y = random.uniform(-0.02, 0.02)
        
        # 构建变换矩阵
        from PIL.Image import AFFINE
        matrix = [
            1/scale, shear_x, 0,
            shear_y, 1/scale, 0,
            0, 0, 1
        ]
        
        return image.transform(image.size, AFFINE, matrix[:6], Image.BICUBIC)
    
    def lens_distortion(self, image: Image.Image) -> Image.Image:
        """镜头畸变（桶形/枕形）"""
        # 使用 OpenCV 实现镜头畸变
        img = np.array(image)
        h, w = img.shape[:2]
        
        # 畸变参数
        k1 = random.uniform(-0.3, 0.3)  # 径向畸变系数
        k2 = random.uniform(-0.1, 0.1)
        
        # 相机矩阵
        fx = fy = max(w, h)
        cx, cy = w / 2, h / 2
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # 畸变系数
        dist_coeffs = np.array([k1, k2, 0, 0, 0])
        
        # 计算新相机矩阵和映射
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 0
        )
        
        # 应用畸变
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
        )
        distorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        
        return Image.fromarray(distorted)
    
    def canvas_expand_trim(self, image: Image.Image) -> Image.Image:
        """画布扩展与边缘裁剪"""
        w, h = image.size
        expand_ratio = random.uniform(0.05, 0.15)  # 扩展5%-15%
        
        # 扩展画布
        new_w = int(w * (1 + expand_ratio))
        new_h = int(h * (1 + expand_ratio))
        
        # 创建新画布（白色背景）
        new_image = Image.new('RGB', (new_w, new_h), (255, 255, 255))
        
        # 随机放置原图
        paste_x = random.randint(0, new_w - w)
        paste_y = random.randint(0, new_h - h)
        new_image.paste(image, (paste_x, paste_y))
        
        # 随机裁剪回原大小
        crop_x = random.randint(0, new_w - w)
        crop_y = random.randint(0, new_h - h)
        new_image = new_image.crop((crop_x, crop_y, crop_x + w, crop_y + h))
        
        return new_image
    
    # ========== 颜色与光照调整 ==========
    
    def color_jitter(self, image: Image.Image) -> Image.Image:
        """颜色抖动（亮度、对比度）"""
        # 亮度
        brightness_factor = random.uniform(0.9, 1.1)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        # 对比度
        contrast_factor = random.uniform(0.9, 1.1)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        return image
    
    def rgb_shift(self, image: Image.Image) -> Image.Image:
        """RGB通道偏移"""
        img_array = np.array(image)
        
        # 每个通道独立偏移 (-5 到 +5)
        shifts = [random.randint(-5, 5) for _ in range(3)]
        
        for i in range(3):
            if shifts[i] != 0:
                img_array[:, :, i] = np.roll(img_array[:, :, i], shifts[i], axis=0)
        
        return Image.fromarray(img_array)
    
    def color_temperature(self, image: Image.Image) -> Image.Image:
        """色温调整"""
        img_array = np.array(image).astype(np.float32)
        
        # 调整色温：暖色（增强红/减弱蓝）或冷色（增强蓝/减弱红）
        if random.random() > 0.5:
            # 暖色
            img_array[:, :, 0] *= 1.1  # 增强红
            img_array[:, :, 2] *= 0.9  # 减弱蓝
        else:
            # 冷色
            img_array[:, :, 0] *= 0.9  # 减弱红
            img_array[:, :, 2] *= 1.1  # 增强蓝
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def gamma_correction(self, image: Image.Image) -> Image.Image:
        """Gamma校正"""
        gamma = random.uniform(0.8, 1.2)
        
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.power(img_array, gamma) * 255
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def channel_dropout(self, image: Image.Image) -> Image.Image:
        """通道随机丢失（随机将某个通道置为0）"""
        img_array = np.array(image)
        
        # 随机选择1个通道置零
        channel = random.randint(0, 2)
        img_array[:, :, channel] = 0
        
        return Image.fromarray(img_array)
    
    # ========== 噪声干扰 ==========
    
    def gaussian_noise(self, image: Image.Image) -> Image.Image:
        """高斯噪声"""
        img_array = np.array(image).astype(np.float32)
        
        # 噪声强度
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, img_array.shape)
        
        noisy = img_array + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy)
    
    def salt_pepper_noise(self, image: Image.Image) -> Image.Image:
        """椒盐噪声"""
        img_array = np.array(image)
        
        # 噪声密度
        prob = random.uniform(0.01, 0.03)
        
        # 生成噪声掩码
        salt_mask = np.random.random(img_array.shape[:2]) < prob / 2
        pepper_mask = np.random.random(img_array.shape[:2]) < prob / 2
        
        # 应用噪声
        img_array[salt_mask] = [255, 255, 255]
        img_array[pepper_mask] = [0, 0, 0]
        
        return Image.fromarray(img_array)
    
    def poisson_noise(self, image: Image.Image) -> Image.Image:
        """泊松噪声"""
        img_array = np.array(image).astype(np.float32)
        
        # 泊松噪声：每个像素值添加泊松分布的随机值
        scale = random.uniform(10, 30)
        noisy = np.random.poisson(img_array / scale) * scale
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy)
    
    def speckle_noise(self, image: Image.Image) -> Image.Image:
        """散斑噪声（乘法噪声）"""
        img_array = np.array(image).astype(np.float32)
        
        # 乘法噪声
        sigma = random.uniform(0.05, 0.15)
        noise = np.random.normal(1, sigma, img_array.shape)
        
        noisy = img_array * noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy)
    
    # ========== 模糊与画质损伤 ==========
    
    def gaussian_blur(self, image: Image.Image) -> Image.Image:
        """高斯模糊"""
        radius = random.uniform(0.5, 1.5)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def motion_blur(self, image: Image.Image) -> Image.Image:
        """运动模糊"""
        # 使用 imgaug 实现运动模糊（如果没有安装，回退到高斯模糊）
        try:
            import imgaug.augmenters as iaa
            img_array = np.array(image)
            
            # 运动模糊核大小
            k = random.randint(5, 15)
            angle = random.uniform(0, 360)
            
            aug = iaa.MotionBlur(k=k, angle=angle)
            blurred = aug(image=img_array)
            
            return Image.fromarray(blurred)
        except ImportError:
            # 回退到高斯模糊
            return self.gaussian_blur(image)
    
    def jpeg_artifacts(self, image: Image.Image) -> Image.Image:
        """JPEG压缩伪影"""
        # 保存为JPEG并重新加载，模拟压缩伪影
        quality = random.randint(75, 85)
        
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        return Image.open(buffer)
    
    # 辅助函数：计算透视变换系数
    @staticmethod
    def find_coeffs(pa, pb):
        """计算透视变换系数"""
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        
        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(pb).reshape(8)
        
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

# ===== 集成到 StreamingHFPreprocessor =====

class AugmentedStreamingHFPreprocessor:
    """带增强的流式预处理器"""
    
    def __init__(self, 
                 hf_dataset_id: str, 
                 split: str = "train",
                 apply_augmentation: bool = True,
                 augmentation_prob: float = 0.8,
                 min_ops: int = 2,
                 max_ops: int = 3):
        
        self.ds = load_dataset(hf_dataset_id, split=split, streaming=False)
        self.apply_augmentation = apply_augmentation
        self.augmentation_prob = augmentation_prob
        
        # 初始化增强器
        self.augmenter = LatexAugmentation(
            apply_prob=augmentation_prob,
            min_ops=min_ops,
            max_ops=max_ops
        )
        
        print(f"Loaded dataset with {len(self.ds)} samples")
        print(f"Augmentation: {min_ops}-{max_ops} ops per image")
    
    def __call__(self, **kwargs):
        for idx, item in enumerate(self.ds):
            image = item['image']
            text = item['text']
            
            # 应用增强（只对训练集）
            if self.apply_augmentation:
                image = self.augmenter(image)
            
            yield {
                "messages": [
                    {
                        "role": "user",
                        "content": "<image>请根据图片中的公式生成对应的 latex 语法正确的公式文本。"
                    },
                    {
                        "role": "assistant",
                        "content": text
                    }
                ],
                "images": [image],
                "__id__": idx
            }

# ===== 使用示例 =====

# 训练集（带增强）
# train_preprocessor = AugmentedStreamingHFPreprocessor(
#     "your-username/latex-formulas",
#     split="train",
#     apply_augmentation=True,
#     augmentation_prob=0.9,  # 90%的图片做增强
#     min_ops=2,               # 最少2种操作
#     max_ops=3                # 最多3种操作
# )