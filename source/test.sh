g++ main.cpp -o app \
  -I/usr/local/include/opencv4 \
  -L/usr/local/lib \
  -lopencv_core \
  -lopencv_imgproc \
  -lopencv_imgcodecs \
  -lopencv_highgui \
  -lopencv_features2d \
  -lopencv_flann
./app

# g++ main.cpp -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui
# ./a.out