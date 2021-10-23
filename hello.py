from flask import Flask, render_template
################################
# 바운딩박스로 칼로리 측정하기 #
################################


#########################################################################################################

import cv2
import numpy as np
import re
from PIL import Image
from matplotlib import pyplot as plt

#########################################################################################################

pred = Image.open('darknet/predictions.jpg')
pred.save("static/images/predictions.jpg")


# 입력 영상 & 템플릿 영상 불러오기
template = cv2.imread('static/images/card.jpg', cv2.IMREAD_COLOR)
test_image = cv2.imread('static/images/test_img.jpg', cv2.IMREAD_COLOR)


# prediction.jpg (이미지)
prediction_img = cv2.imread('static/images/predictions.jpg')  # 변경
# prediction.txt (텍스트) & data 파일
prediction_txt = 'darknet/prediction_food19.txt'  # 변경
data_path = 'darknet/data/food19/food19_kcal.data'

#########################################################################################################


test_img = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# 입력 영상 밝기 50증가, 가우시안 잡음(sigma=10) 추가하여 약간의 변형을 줌
noise = np.zeros(test_img.shape, np.int32)

# cv2.randn은 가우시안 형태의 랜덤 넘버를 지정, 노이즈 영상에 평균이 50 시그마 10인 노이즈 추가
cv2.randn(noise,50,10)

# 노이즈를 입력 영상에 더함, 원래 영상보다 50정도 밝아지고 시그마 10정도 변형
test_img = cv2.add(test_img, noise, dtype=cv2.CV_8UC3)

# 탬플릿 매칭 & 결과 분석
res = cv2.matchTemplate(test_img, template, cv2.TM_CCOEFF_NORMED) # 여기서 최댓값 찾기

# 최솟값 0, 최댓값 255 지정하여 결과값을 그레이스케일 영상으로 만들기
res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# 최댓값을 찾아야하므로 minmaxloc 사용, min, max, min좌표, max좌표 반환
_, maxv, _, maxloc = cv2.minMaxLoc(res)

# 탬플릿에 해당하는 영상이 입력 영상에 없으면 고만고만한 값에서 가장 큰 값을 도출.
# 그래서 maxv를 임계값 0.7 or 0.6을 설정하여 템플릿 영상이 입력 영상에 존재하는지 파악
# print('maxv : ', maxv)
# print('maxloc : ', maxloc) 

# 매칭 결과를 빨간색 사각형으로 표시
# maxv가 어느 값 이상이여야지 잘 찾았다고 간주할 수 있다.
template_height, template_width = template.shape[:2]
dst = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
cv2.rectangle(dst, maxloc, (maxloc[0] + template_width, maxloc[1] + template_height), (0, 0, 255), 20)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# print('template_height:', template_height)
# print('template_width:', template_width)

# 결과 영상 화면 출력
# plt.imshow(res_norm)
# plt.title('res_norm')
# plt.xticks([])
# plt.yticks([])
# plt.show()

# plt.imshow(dst)
# plt.title('template matching')
# plt.xticks([])
# plt.yticks([])
# plt.show()

print('({},{})'.format(template_height, template_width))

#########################################################################################################

# 바운딩박스 크기로 칼로리 측정하기1

# result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
# plt.imshow(result_img_rgb)
# plt.title('result_img_rgb')
# plt.xticks([])
# plt.yticks([])
# plt.show()



with open(prediction_txt, "r", encoding="utf-8") as txt:
  txt_lines = txt.readlines()
  txt_line = txt_lines[2]
  # Bounding Box : Left=427, Top=903, Right=1724, Bottom=2112
  txt_result = re.findall("\d+", txt_line)
  box_left = int(txt_result[0])
  box_right = int(txt_result[2])
  box_top = int(txt_result[1])
  box_bottom = int(txt_result[3])
  print(box_left,box_right,box_top,box_bottom)
  
  width = box_right - box_left
  height = box_bottom - box_top
  print('width:',width)
  print('height:',height)

  txt.close()

print('================\n')
card_pixel = template_height
card_length = 8.56 #(cm)

food_width = card_length * (width/card_pixel)
food_height = card_length * (height/card_pixel)
food_area = food_width * food_height
print('food_witdth', food_width)
print('food_height', food_height)
print('food_area', food_area)

# 식빵의 가로 = 11, 세로 = 9 일 때 칼로리 600kcal
toast_area = 99
toast_kcal = 400

kcal = (toast_kcal * food_area)/toast_area
b = round(kcal,2)

print('kcal:',kcal)

print('\n \n')
print('=========================================')
print('=============== 최종 결과 ===============')
print('=========================================')
print('\n \n')

# prediction 이미지

prediction_img_rgb = cv2.cvtColor(prediction_img, cv2.COLOR_BGR2RGB)
# plt.imshow(prediction_img_rgb)
# plt.title('Prediction')
# plt.xticks([])
# plt.yticks([])
# plt.show()

with open(prediction_txt, 'r', encoding="utf-8") as pf, open(data_path, 'r', encoding="utf-8") as df:
  lines = pf.readlines()
  food_count= len(lines) - 1
  # print('감지된 음식의 수:', food_count)
  
  if (food_count == 0):
    print("음식이 감지되지 않았습니다.")
  
  else:
      prediction = lines[1:]
      prediction_length = len(prediction)
      prediction = prediction[0::3]
      df_lines = df.readlines()

      for i in prediction:
        food_name = i.split(":")[0]
        
        for lines in df_lines:
          data_names = lines.split(" ")
          data_name = data_names[1]
          
          if food_name == data_name:
            # result=[]

            a = data_names[3]
            # b = data_names[2]
            # print('{}({:.1f}kcal)'.format(result[0],kcal))

  pf.close()
  df.close()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("image.html", a=a, b=b)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)