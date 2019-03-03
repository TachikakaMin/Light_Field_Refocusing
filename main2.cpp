#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <random>
using namespace cv;
using namespace std;
const int camera_H = 16;
const int camera_W = 16;
const int camera_Tot = camera_W*camera_H;
String image_Name[camera_Tot+10]; // by default
Mat_<cv::Vec3d> image[camera_Tot+10];
Size2d camera_Main_Pos(0.52,0.52);
Mat_<cv::Vec3d> image_Final;
double f = 1;
double z1 = 160;
double z2 = 35;

bool read_Image()
{
    for (int i = 0;i<=camera_Tot;i++)
    {
        if (i<10) image_Name[i] = "./toyLF/lowtoys00"+std::to_string(i)+".bmp";
        else if (i<100) image_Name[i] = "./toyLF/lowtoys0"+std::to_string(i)+".bmp";
        else if (i<1000) image_Name[i] = "./toyLF/lowtoys"+std::to_string(i)+".bmp";
    }
    Mat readimg;
    for (int i=1;i<=camera_Tot;i++) 
    {
        readimg = imread( image_Name[i]); // Read the file
        image[i] = readimg;
        if( image[i].empty() )                      // Check for invalid input
        {
            cout <<  "Could not open or find the image: " << image_Name[i] << std::endl ;
            return 0;
        }
    }
    return 1;
    // namedWindow( "test window", WINDOW_AUTOSIZE ); // Create a window for display.
    // imshow( "test window", image[1] ); 
}

void naive_Image()
{
    image_Final = Mat_<cv::Vec3d>(image[1].size());
    int height = image_Final.size().height;
    int width  = image_Final.size().width;

    double y = 1+camera_Main_Pos.height*(camera_H-1);
    double x = 1+camera_Main_Pos.width*(camera_W-1);
    int x1 = floor(x), x2 = ceil(x);
    int y1 = floor(y), y2 = ceil(y);
    printf("%d %d %d %d\n",x1,x2,y1,y2);

    for (int i=0;i<height;i++)
        for (int j=0;j<width;j++)
        {
            cv::Vec3d a;
            if (x1!=x2) a = abs(x2-x)*image[y1*camera_W+x1](i,j)+abs(x1-x)*image[y1*camera_W+x2](i,j);
                else a = image[y1*camera_W+x1](i,j);
            cv::Vec3d b;
            if (x1!=x2) b = abs(x2-x)*image[y2*camera_W+x1](i,j)+abs(x1-x)*image[y2*camera_W+x2](i,j);
                else b = image[y2*camera_W+x1](i,j);
            if (y1!=y2) image_Final(i,j) = abs(y2-y)*a+abs(y1-y)*b;
                else image_Final(i,j) = a;
            image_Final(i,j)/=255.0;
        }
}

template <typename T>
T get_Normal_pdf(T x, T m, T s)
{
    static const T inv_sqrt_2pi = 0.3989422804014327;
    T a = (x - m) / s;

    return inv_sqrt_2pi / s * std::exp(-T(0.5) * a * a);
}



void advance_Image()
{
    image_Final = Mat_<cv::Vec3d>(image[1].size());
    int height = image_Final.size().height;
    int width  = image_Final.size().width;

    double y = 1+camera_Main_Pos.height*(camera_H-1);
    double x = 1+camera_Main_Pos.width*(camera_W-1);
    int dx,dy;
    vector<double> weight;
    vector<int> point;
    weight.clear();
    point.clear();

    for (int a=1;a<=camera_W;a++)
        for (int b=1;b<=camera_H;b++)
        {
            int num = (b-1)*camera_W+a;
            double dis = (x - a) * (x - a);
            dis += (y - b) * (y - b);
            dis = sqrt(dis);
            double w = get_Normal_pdf(dis,0.0,2.0);
            if (w>0.01) 
            {
                weight.push_back(w);
                dx = (a-x)*f*width*4.4/z1;
                dy = -(b-y)*f*height*6.3/z1;
                point.push_back(num);
                point.push_back(dx);
                point.push_back(dy);
            }
        }
    if (weight.empty()) return ;
    printf("size: %d\n",weight.size());
    double sum = 0;
    vector<double> new_weight;
    new_weight.clear();

    for (int i=0;i<height;i++)
        for (int j=0;j<width;j++)
        {
            sum = 0;
            new_weight.clear();
            for (int a = 0;a<weight.size();a++)
            {
                int num = point[a*3];
                dx = point[a*3+1];
                dy = point[a*3+2];
                if (i+dy >= 0 && i+dy < height)
                    if (j+dx >=0 && j+dx < width)
                        sum+=weight[a];
            }
            if (sum <= 0.00000001) continue;
            for (int a = 0;a<weight.size();a++){
                new_weight.push_back(weight[a]/sum);
            }
            for (int a = 0;a<weight.size();a++)
            {
                int num = point[a*3];
                dx = point[a*3+1];
                dy = point[a*3+2];
                if (i+dy >= 0 && i+dy < height)
                    if (j+dx >=0 && j+dx < width)
                        image_Final(i,j) += new_weight[a]*image[num](i+dy,j+dx);
            }
            image_Final(i,j)/=255.0;
        }

    return ;
}


int main( int argc, char** argv )
{
    if (!read_Image()) return 0;

    // naive_Image();
    advance_Image();
    double dz = 1;
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image_Final );                // Show our image inside it.
    while (1)
    {
        int o = waitKey(10);
        if (o == 'q') return 0;
        if (o==0) z2+=dz;
        if (o==1) z2-=dz;
        if (o==2) z1-=dz;
        if (o==3) z1+=dz;
        if (z1==0 || z2==0) break;
        if (o>=0 && o<=3) {printf("z1: %.3lf  z2: %.3lf\n",z1,z2);advance_Image();imshow( "Display window", image_Final );} 
    }

    // namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    // imshow( "Display window", image_Final );                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}