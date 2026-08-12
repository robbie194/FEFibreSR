import cv2


def main():
    # 打开一个视频文件
    cap = cv2.VideoCapture('../input/tennis.mov')
    # 检查 cap 对象是否具有 frame_height 属性
    if cap.get(cv2.CAP_PROP_FRAME_HEIGHT):
        print("cap 对象具有 frame_height 属性")
        # 获取视频的帧高
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"视频帧高: {frame_height}")
    else:
        print("cap 对象没有 frame_height 属性")
    # 释放资源
    cap.release()


if __name__ == "__main__":
    main()
    # a = None
    # if not a:
    #     print(1)