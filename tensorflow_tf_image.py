Tensorflow中内置的图像处理的函数
	主要使用的模块就是tf.image
	肯定没有OpenCV那么多那么强大啦，但是仅仅是作为简单的预处理的话，完全是够用了。

数据类型转化
	convert_image_dtype(image,dtype,saturate=False,name=None)
		作用：把图片元素类型，转成想要的类型，返回转换后的图片,注意，要是转成了float类型之后，像素值会在 [0,1)这个范围内。 
		参数: 
		image: 图像 
		dtype: 待转换类型 
		saturate: If True, clip the input before casting (if necessary). 
		name: 可选操作名


图片的放大缩小
	tf.image.resize_images()
	参数: 
		images: 4维tensor,形状为 [batch, height, width, channels] 或者3维tensor,形状为 [height, width, channels]. 
		size: 1维 int32类型的 Tensor,包含两个元素:new_height, new_width. 
		method: 改变形状的方法,默认是ResizeMethod.BILINEAR.
		Method	图像大小的调整算法	值
		0 双线性插值法	tf.image.ResizeMethod.BILINEAR
		1 最近邻居法	tf.image.ResizeMethod.NEAREST_NEIGHBOR
		2 双三次插值法	tf.image.ResizeMethod.BICUBIC
		3 面积插值法	tf.image.ResizeMethod.AREA

		在使用TensorFlow进行图片的放大缩小时，有三种方式： 
		tf.image.resize_nearest_neighbor（）:临界点插值 
		tf.image.resize_bilinear(）：双线性插值 
		tf.image.resize_bicubic(）：双立方插值算法
		tf.image.resize_area(...): 应用区域插值调整图像尺寸。


	resize_image_with_crop_or_pad(…): Crops and/or pads an image to a target width and height. 
	central_crop(…): Crop the central region of the image. 
	crop_and_resize(…): Extracts crops from the input image tensor and bilinearly resizes them (possibly 
	crop_to_bounding_box(…): Crops an image to a specified bounding box.

	
图像翻转
	flip_left_right(…): 左右翻转 
		作用:左右翻转一幅图片,返回一个形状和原图片相同的图片(翻转后) 
		参数: 
			image: 3维tensor,形状为[height, width, channels].
	flip_up_down(…): 上下翻转 
	transpose_image(…): 对角线翻转 
	random_flip_left_right(…): 随机左右翻转 
	random_flip_up_down(…): 随机上下翻转

三.颜色变换
	和前面使用翻转可以来”增加数据集”以外,调节颜色属性也是同样很有用的方法,这里主要有调整亮度,对比度,饱和度,色调等方法.如下: 
	亮度:
		adjust_brightness(…): 调整亮度
			作用:调节亮度 
			参数: 
			image: tensor,原图片 
			delta: 标量,待加到像素值上面的值.
		random_brightness(…): 随机调整亮度

	对比度:
		adjust_contrast(…): 调整对比度 
		random_contrast(…): 随机调整亮度

	饱和度:
		adjust_saturation(…): 调整饱和度 
		random_saturation(…): 随机调整饱和度

	色调:
		adjust_hue(…): 调整色调 
		random_hue(…): 随机调整色调

	这里只举一个调节亮度的例子,其他大同小异,可以试一下看结果 
	adjust_brightness(image,delta)

	作用:调节亮度 
	参数: 
	image: tensor,原图片 
	delta: 标量,待加到像素值上面的值.

数据增强相关
	数据增强的作用就不多说了，tensorflow的数据预处理部分也给出了一些数据增强的方法，表面上看上去也许只是普通的图片变换，但是这些方法在一方面来说能够非常有效的扩充“数据集”。 
	这几个方法就是tf.image里面带有random的一些方法。也就是随机怎样怎样，上面其实已经列国了，这里再列出来一遍,大家可以根据需要来选择可能用到的.

	random_brightness(…): Adjust the brightness of images by a random factor.

	random_contrast(…): Adjust the contrast of an image by a random factor.

	random_flip_left_right(…): Randomly flip an image horizontally (left to right).

	random_flip_up_down(…): Randomly flips an image vertically (upside down).

	random_hue(…): Adjust the hue of an RGB image by a random factor.

	random_saturation(…): Adjust the saturation of an RGB image by a random factor.

	tf.image.sample_distorted_bounding_box 是随机截取图像。

图片的标准化
	在使用TensorFlow对图像数据进行训练之前，常需要执行图像的标准化操作，它与归一化是有区别的，归一化不改变图像的直方图，标准化操作会改变图像的直方图。
	标准化操作使用如下函数： 
	tf.image.per_image_standardization（）

# 重新调整大小。
resized = tf.image.resize_images(image_data,[300,300],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # resize_images(images,size,method=ResizeMethod.BILINEAR,align_corners=False)'
    # 参数如下：
    # images:需要调整的图片，通常是一个tensor
    # size：调整之后的大小，一般是一个长度为2的list
    # method：调整大小使用的方法，这里我们使用最近邻居法。
    # align_corner：是否对齐四角。
	
# 调整大小，采用切割或者填充的方式。
cropped = tf.image.resize_image_with_crop_or_pad(image_data,300,300)
    # 如果原始图像的尺寸大于目标图像，那么这个函数会自动切割原始图像中的居中的部分。
    # 如果原始图像的尺寸小于目标图像，那么这个函数会自动在原始图像的周围采用全0填充。
    # resize_image_with_crop_or_pad(image, target_height, target_width)
    # image：待调整的图像。
    # target_height：目标图像的高度。
    # target_width：目标图像的宽度。
	

	



# Randomly crop a [height, width] section of the image.
distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

# Randomly flip the image horizontally.
distorted_image = tf.image.random_flip_left_right(distorted_image)

# Because these operations are not commutative, consider randomizing
# the order their operation.
# NOTE: since per_image_standardization zeros the mean and makes
# the stddev unit, this likely has no effect see tensorflow#1458.
distorted_image = tf.image.random_brightness(distorted_image,
											 max_delta=63)
distorted_image = tf.image.random_contrast(distorted_image,
										   lower=0.2, upper=1.8)

# Subtract off the mean and divide by the variance of the pixels.
float_image = tf.image.per_image_standardization(distorted_image)