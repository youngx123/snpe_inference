
#include "face_landmark.h"
#include <fstream>
// #include <math>
FaceLandMarker::FaceLandMarker()
{
}

FaceLandMarker::~FaceLandMarker()
{
	outputs_host.clear();
}

inline int sign(float x)
{
	if (x == 0)
	{
		return 0;
	}
	else if (x > 0)
	{
		return 1;
	}
	else
	{
		return -1;
	}
}

bool FaceLandMarker::Init(int dw, int dh, const std::string &model_path, float score_thresh,
						  std::vector<std::string> &output_names, int infer_device, size_t batch_size)

{
	dst_w = dw;
	dst_h = dh;

	infer_ = InferenceFactory::CreateInference();
	printf("load %s model  \n", model_path.c_str());
	InferDevice device = InferDevice::kCPU;

	if (infer_device == 1)
	{
		device = InferDevice::kIGPU;
	}
	else if (infer_device == 2)
	{
		device = InferDevice::kNPU;
	}
	// infer->Init();
	if (!infer_->Init(model_path, batch_size, device, output_names, false, false))
	{
		printf("init  %s  model failed\n", model_path.c_str());
		return 0;
	}
	Batch_size = batch_size;
	score_threshold = score_thresh;
	return true;
}

void FaceLandMarker::PrePrecess(cv::Mat &image)
{
	cv::resize(image, image, cv::Size(dst_w, dst_h), cv::INTER_AREA);
}

void FaceLandMarker::Inference(cv::Mat &image, std::string save_file)
{
	src_img = image.clone();
	// 1. pre-processing
	PrePrecess(image);
	// 2. model inference
	infer_->DoInference(image);
	// get output results from model inference
	// outputs_host = infer_->outputs_host;
	outputs_host = infer_->outputs_host_;
	// 3. decoder
	decode();
	draw(save_file);
}

void FaceLandMarker::save_text(std::ofstream &outfile, std::string index)
{
	// ofstream outfile;
	// outfile.open("pts_result.txt");
	for (int k = 0; k < 17; k++)
	{
		landmark_with_score tmp = face_pts[k];
		outfile << index << " " << tmp.score << " " << tmp.landmark.x << " " << tmp.landmark.y << "\n";
	}
}

void FaceLandMarker::decode2()
{
	face_pts.reserve(17);
	face_pts.clear();
	// 输出tensor 维度为：(1,64,64,17)
	cv::Mat feact_mat(64, 64, CV_32FC(17));
	memcpy(feact_mat.data, outputs_host[0], 64 * 64 * 17 * sizeof(float));
	std::cout << "channel = " << feact_mat.channels() << std::endl; // 输出为5

	std::vector<cv::Mat> heatMap;
	cv::split(feact_mat, heatMap);
	for (int j = 0; j < 17; j++) // 17 * 1
	{
		// memcpy(feact_mat.data, outputs_face, 64 * 64 * sizeof(float));
		cv::Mat channel_map = heatMap.at(j);
		double minVal = 0., maxVal = 0.0;
		int minIdx[2] = {}, maxIdx[2] = {}; // minnimum Index, maximum Index
		cv::minMaxIdx(channel_map, &minVal, &maxVal, minIdx, maxIdx);

		cv::Point2f pts; // 在 256*256中的坐标
		pts.x = maxIdx[1];
		pts.y = maxIdx[0];
		// diff = np.array([
		//                     heatmap[py][px + 1] - heatmap[py][px - 1],
		//                     heatmap[py + 1][px] - heatmap[py - 1][px]
		//                 ])
		// float diff_x = channel_map.at<float>(pts.y, pts.x + 1) - channel_map.at<float>(pts.y, pts.x - 1);
		// float diff_y = channel_map.at<float>(pts.y + 1, pts.x) - channel_map.at<float>(pts.y - 1, pts.x);

		float diff_x = channel_map.at<float>(pts.x, pts.y + 1) - channel_map.at<float>(pts.x, pts.y - 1);
		float diff_y = channel_map.at<float>(pts.x + 1, pts.y) - channel_map.at<float>(pts.x - 1, pts.y);

		float offset_x = sign(diff_x) * 0.25;
		float offset_y = sign(diff_y) * 0.25;

		pts.x = (pts.x + offset_x) * 4.0;
		pts.y = (pts.y + offset_y) * 4.0;
		printf("%f, %f\n", offset_x, offset_y);
		printf("id = %d, %f , x=%f, y=%f\n", j, maxVal, pts.x, pts.y);
		landmark_with_score tmp{pts, maxVal};
		face_pts.push_back(tmp);
		// outputs_face += 64 * 64;
	}
}

void FaceLandMarker::decode3()
{
	outputs_face = reinterpret_cast<float *>(outputs_host[0]);
	face_pts.reserve(17);
	face_pts.clear();
	// 输出tensor 维度为：(1,64,64,17)
	cv::Mat feact_mat(64, 64, CV_32FC1);
	// memcpy(feact_mat.data, outputs_host[0], 64 * 64 * 17 * sizeof(float));
	std::cout << "channel = " << feact_mat.channels() << std::endl; // 输出为5

	std::vector<cv::Mat> heatMap;
	cv::split(feact_mat, heatMap);
	for (int j = 0; j < 17; j++) // 17 * 1
	{
		memcpy(feact_mat.data, outputs_face, 64 * 64 * sizeof(float));
		// cv::Mat channel_map = heatMap.at(j);
		double minVal = 0., maxVal = 0.0;
		int minIdx[2] = {}, maxIdx[2] = {}; // minnimum Index, maximum Index
		cv::minMaxIdx(feact_mat, &minVal, &maxVal, minIdx, maxIdx);

		cv::Point2f pts; // 在 256*256中的坐标
		pts.x = maxIdx[1] * 4.0f;
		pts.y = maxIdx[0] * 4.0f;
		printf("id = %d, %f , x=%f, y=%f\n", j, maxVal, pts.x, pts.y);
		landmark_with_score tmp{pts, maxVal};
		face_pts.push_back(tmp);
		outputs_face += 64 * 64;
	}
}

void FaceLandMarker::decode()
{
	face_pts.reserve(17);
	face_pts.clear();
	// 输出tensor 维度为：(1,64,64,17)
	cv::Mat feact_mat(64, 64, CV_32FC(17));
	memcpy(feact_mat.data, outputs_host[0], 64 * 64 * 17 * sizeof(float));
	std::cout << "channel = " << feact_mat.channels() << std::endl; // 输出为5

	std::vector<cv::Mat> heatMap;
	cv::split(feact_mat, heatMap);
	for (int j = 0; j < 17; j++) // 17 * 1
	{
		// memcpy(feact_mat.data, outputs_face, 64 * 64 * sizeof(float));
		cv::Mat channel_map = heatMap.at(j);
		double minVal = 0., maxVal = 0.0;
		int minIdx[2] = {}, maxIdx[2] = {}; // minnimum Index, maximum Index
		cv::minMaxIdx(channel_map, &minVal, &maxVal, minIdx, maxIdx);

		cv::Point2f pts; // 在 256*256中的坐标
		pts.x = maxIdx[1] * 4.0f;
		pts.y = maxIdx[0] * 4.0f;
		printf("id = %d, %f , x=%f, y=%f\n", j, maxVal, pts.x, pts.y);
		landmark_with_score tmp{pts, maxVal};
		face_pts.push_back(tmp);
		// outputs_face += 64 * 64;
	}
}

void FaceLandMarker::draw(std::string save_file)
{
	// auto left_eye_corner_1 = my_face_landmarks.landmarks[1].score;// 左眼左侧眼角
	cv::circle(src_img, face_pts[1].landmark, 2, cv::Scalar(5, 255, 0));
	char show_text[256];
	sprintf(show_text, "le1:%.3f x:%.1f y: %.1f", face_pts[1].score, face_pts[1].landmark.x, face_pts[1].landmark.y);
	cv::putText(src_img, show_text, cv::Point(2, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 5), 1);

	cv::circle(src_img, face_pts[2].landmark, 2, cv::Scalar(5, 255, 0));
	sprintf(show_text, "le2:%.3f x:%.1f y: %.1f", face_pts[2].score, face_pts[2].landmark.x, face_pts[2].landmark.y);
	cv::putText(src_img, show_text, cv::Point(2, 35), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 5), 1);

	cv::circle(src_img, face_pts[5].landmark, 2, cv::Scalar(5, 255, 0));
	sprintf(show_text, "le5:%.3f x:%.1f y: %.1f", face_pts[5].score, face_pts[5].landmark.x, face_pts[5].landmark.y);
	cv::putText(src_img, show_text, cv::Point(2, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 5), 1);

	// auto left_eye_corner_2 = my_face_landmarks.landmarks[4].score;// 左眼右侧眼角
	cv::circle(src_img, face_pts[4].landmark, 2, cv::Scalar(5, 255, 0));
	sprintf(show_text, "le4:%.3f x:%.1f y:%.1f", face_pts[4].score, face_pts[4].landmark.x, face_pts[4].landmark.y);
	cv::putText(src_img, show_text, cv::Point(2, 75), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 5), 1);

	// auto right_eye_corner_1 = my_face_landmarks.landmarks[7].score;// 右眼左侧眼角
	cv::circle(src_img, face_pts[7].landmark, 2, cv::Scalar(5, 255, 0));
	sprintf(show_text, "re7:%.3f x:%.1f y:%.1f", face_pts[7].score, face_pts[7].landmark.x, face_pts[7].landmark.y);
	cv::putText(src_img, show_text, cv::Point(2, 95), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 5), 1);

	cv::circle(src_img, face_pts[9].landmark, 2, cv::Scalar(5, 255, 0));
	sprintf(show_text, "re9:%.3f x:%.1f y:%.1f", face_pts[9].score, face_pts[9].landmark.x, face_pts[9].landmark.y);
	cv::putText(src_img, show_text, cv::Point(2, 115), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 5), 1);

	cv::circle(src_img, face_pts[11].landmark, 2, cv::Scalar(5, 255, 0));
	sprintf(show_text, "re11:%.3f x:%.1f y:%.1f", face_pts[11].score, face_pts[11].landmark.x, face_pts[11].landmark.y);
	cv::putText(src_img, show_text, cv::Point(2, 135), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 5), 1);

	// auto right_eye_corner_2 = my_face_landmarks.landmarks[10].score;// 右眼右侧眼角
	cv::circle(src_img, face_pts[10].landmark, 2, cv::Scalar(5, 255, 0));
	sprintf(show_text, "r10:%.3f x:%.1f y:%.1f", face_pts[10].score, face_pts[10].landmark.x, face_pts[10].landmark.y);
	cv::putText(src_img, show_text, cv::Point(2, 155), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 5), 1);

	// auto nose_tip_score = my_face_landmarks.landmarks[0].score;// 鼻尖
	sprintf(show_text, "nti:%.3f x:%.1f y:%.1f", face_pts[0].score, face_pts[0].landmark.x, face_pts[0].landmark.y);
	cv::circle(src_img, face_pts[0].landmark, 2, cv::Scalar(0, 255, 255)); // nose_tip
	cv::putText(src_img, show_text, cv::Point(2, 175), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 5), 1);

	// auto left_mouth_corner_1 = my_face_landmarks.landmarks[13].score;// 左嘴角
	cv::circle(src_img, face_pts[13].landmark, 2, cv::Scalar(255, 5, 255));
	sprintf(show_text, "m1:%.3f x:%.1f y:%.1f", face_pts[13].score, face_pts[13].landmark.x, face_pts[13].landmark.y);
	cv::putText(src_img, show_text, cv::Point(2, 195), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 5), 1);

	// auto left_mouth_corner_2 = my_face_landmarks.landmarks[14].score;// 右嘴角
	cv::circle(src_img, face_pts[14].landmark, 2, cv::Scalar(255, 5, 255));
	sprintf(show_text, "m2:%.3f x:%.1f y:%.1f", face_pts[14].score, face_pts[14].landmark.x, face_pts[14].landmark.y);
	cv::putText(src_img, show_text, cv::Point(2, 210), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 5), 1);
	if (save_file != "None")
	{
		cv::imwrite(save_file, src_img);
	}
}
