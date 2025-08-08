rm ./app

g++ -std=c++17 custom.cpp reconstruction.cpp  SemiGlobalMatching.cpp -o app  $(pkg-config --cflags --libs opencv4) -O3 -march=native -funroll-loops -fopenmp
# g++ -std=c++17 opencv.cpp -o app  $(pkg-config --cflags --libs opencv4) -O3 -march=native -funroll-loops -fopenmp

./app sift