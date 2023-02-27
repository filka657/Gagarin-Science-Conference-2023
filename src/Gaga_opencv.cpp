//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
//#include <exception>
//#include <memory>
//#include <iostream>
//#include <fstream>
//#include <vector>
//
//using namespace cv;
//using namespace std;
//
//
//
//Mat format_yolov5(const cv::Mat& source) {
//
//	// put the image in a square big enough
//	int col = source.cols;
//	int row = source.rows;
//	int _max = MAX(col, row);
//	Mat resized = cv::Mat::zeros(_max, _max, CV_8UC3);
//	source.copyTo(resized(cv::Rect(0, 0, col, row)));
//
//	// resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
//	Mat result;
//	dnn::blobFromImage(source, result, 1. / 255., cv::Size(640, 640), cv::Scalar(), true, false);
//
//	return result;
//}
//
//int main()
//{
//	auto net = cv::dnn::readNet("yolov5s.onnx");
//
//	std::vector<cv::Mat> predictions;
//	net.forward(predictions, net.getUnconnectedOutLayersNames());
//	const cv::Mat& output = predictions[0];
//
//	VideoCapture cap(0);
//	if (!cap.isOpened()) {
//		cout << "Error opening video stream or file" << endl;
//		return -1;
//	}
//	int i = 0;
//
//	while (1)
//	{
//
//		Mat frame;
//		// Capture frame-by-frame
//		cap >> frame;
//
//		// If the frame is empty, break immediately
//		if (frame.empty())
//			break;
//
//		std::cerr << "frame " << i++ << std::endl;
//		// Display the resulting frame
//		imshow( "Frame", frame );
//
//		// Press  ESC on keyboard to exit
//		char c = waitKey(33);
//		if (c == 27)
//		{
//			destroyWindow("Video");
//			break;
//		}
//	}
//
//
//	// When everything done, release the video capture object
//	cap.release();
//
//	// Closes all the frames
//	destroyAllWindows();
//
//	return 0;
//}
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#define INPUT_WIDTH 640.0
#define INPUT_HEIGHT 640.0
#define SCORE_THRESHOLD 0.2
#define NMS_THRESHOLD 0.4
#define CONFIDENCE_THRESHOLD 0.4

using namespace std;
using namespace cv;

const vector<Scalar> colors = { Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0) };

struct Detection
{
    int class_id;
    float confidence;
    Rect box;
};

vector<string> load_class_list();

void load_net(dnn::Net& net, bool is_cuda);

Mat format_yolov5(const Mat& source);

void detect(Mat& image, dnn::Net& net, vector<Detection>& output, const vector<string>& className);

int main(int argc, char** argv)
{

    vector<string> class_list = load_class_list();

    Mat frame;
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file\n";
        return -1;
    }

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

    dnn::Net net;
    load_net(net, is_cuda);

    auto start = chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    while (true)
    {

        cap >> frame;
        if (frame.empty())
        {
            cout << "End of stream\n";
            break;
        }

        vector<Detection> output;
        detect(frame, net, output, class_list);

        frame_count++;
        total_frames++;

        int detections = output.size();

        for (int i = 0; i < detections; ++i)
        {

            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            rectangle(frame, box, color, 3);

            rectangle(frame, Point(box.x, box.y - 20), Point(box.x + box.width, box.y), color, FILLED);
            putText(frame, class_list[classId].c_str(), Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
        }

        if (frame_count >= 30)
        {

            auto end = chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / chrono::duration_cast<chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = chrono::high_resolution_clock::now();
        }

        if (fps > 0)
        {

            ostringstream fps_label;
            fps_label << fixed << setprecision(2);
            fps_label << "FPS: " << fps;
            string fps_label_str = fps_label.str();

            putText(frame, fps_label_str.c_str(), Point(10, 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }

        imshow("Output", frame);

        if (waitKey(1) != -1)
        {
            cap.release();
            std::cout << "Finished by user\n";
            break;
        }
    }

    cout << "Total frames: " << total_frames << "\n";

    // Closes all the frames
    destroyAllWindows();

    return 0;
}

vector<string> load_class_list()
{
    vector<string> class_list;
    ifstream ifs("classes.txt");
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(dnn::Net& net, bool is_cuda)
{
    auto result = dnn::readNet("yolov5s.onnx");
    if (is_cuda)
    {
        cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        cout << "Running on CPU\n";
        result.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(dnn::DNN_TARGET_CPU);
    }
    net = result;
}

Mat format_yolov5(const Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

void detect(Mat& image, dnn::Net& net, vector<Detection>& output, const vector<string>& className) {
    Mat blob;

    auto input_image = format_yolov5(image);

    dnn::blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    net.setInput(blob);
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float* classes_scores = data + 5;
            Mat scores(1, className.size(), CV_32FC1, classes_scores);
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(Rect(left, top, width, height));
            }

        }

        data += 85;

    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}