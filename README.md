# production-repo

This object tracking model leverages YOLOv7, a powerful object detection model, as its foundation for real-time object detection. It enhances tracking accuracy and efficiency by integrating ByteTrack, a cutting-edge tracking algorithm. This combination of YOLOv7 for precise object identification and ByteTrack for robust tracking ensures seamless and reliable tracking of objects in dynamic environments, making it a formidable solution for various applications, especially for surveillance systems and analytical purposes.

## Demo

<a href="http://www.youtube.com/watch?feature=player_embedded&v=64Ys7Xj4Iho" target="_blank">
 <img src="http://img.youtube.com/vi/64Ys7Xj4Iho/maxresdefault.jpg" alt="Watch the video" width="1280" height="640" border="23" />
</a>

## Installation

Docker
<details><summary> <b>Expand</b> </summary>
  
```shell
docker pull thestralmoon/yolov7_object_tracker:first_commit

docker run --name yolov7_tracker -t -d thestralmoon/yolov7_object_tracker:first_commit

docker exec -it yolov7_tracker bash 
```

</details>
