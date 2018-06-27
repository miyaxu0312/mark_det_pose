# 调用线上人脸检测及人脸关键点标记API为拍摄的视频标记人脸框、关键点、姿态角

## 使用方法：

在 mark_det_pose.py 中填上更改以下信息：

1. bucket鉴权密钥及对应的bucket_name  

upload_access_key = "******************"  

upload_secret_key = "******************"  

bucket_name = "Your bucket name"  

bucket_url = "your bucket url like http://pargr4az5.bkt.clouddn.com/"  

2. 人脸检测及关键点标记的鉴权密钥

process_access_key = "******************"

process_secret_key = "******************"

3. 执行命令

python2 mark_det_pose.py -d <Dir to save data>  

如果需要停止采集数据，在图像显示界面按“q”退出采集数据，已经采集的图像帧将会继续上传并处理

## 特殊依赖：

qiniu -> pip install qiniu

**TIP：本demo需要连上公司VPN方能使用**