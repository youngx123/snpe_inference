#pragma once
// #include <experimental/filesystem>
#include "infer/inference_factory.h"
// #include "infer/model/snpe_infer.h"
#include <vector>
// namespace fs = std::experimental::filesystem; // 修改

typedef struct landmark_with_score
{
	cv::Point2f landmark;
	float score;

} landmark_with_score;

typedef struct face_landmarks
{
	std::vector<landmark_with_score> landmarks;
	float face_score;
	float yaw;
	float pitch;
	bool is_valid_face; //
	bool is_side_face;	// 侧脸 ,临时输出

} face_landmarks;

class FaceLandMarker
{
public:
	FaceLandMarker();
	~FaceLandMarker();

	bool Init(int dw, int dh, const std::string &model_path, float score_thresh,
			  std::vector<std::string> &output_names, int infer_device = 0, size_t batch_size = 1);

	void Inference(cv::Mat &image, std::string save_file = "None");
	void PrePrecess(cv::Mat &image);
	void decode();
	void decode2();
	void decode3();
	void draw(std::string save_file = "None");
	void save_text(std::ofstream &outfile, std::string index);

public:
	cv::Mat src_img;
	std::string model_path_;
	float *outputs = nullptr;
	std::vector<landmark_with_score> face_pts;

private:
	int dst_w;
	int dst_h;
	int Batch_size = 1;
	float score_threshold = 0;
	float NMS_threshold = 0;

private:
	std::vector<void *> outputs_host;
	float *outputs_face;
	InferenceHandle infer_;
};