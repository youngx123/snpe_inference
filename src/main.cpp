/*
#Author      : xyoung
#Date        : 2024-01-25 09:20:53
#LastEditors : fuck c/c++
#LastEditTime: 2024-02-20 13:40:44
*/

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
// #include <experimental/filesystem>
#include <unistd.h>
// #include "./infer/model/snpe_infer.h"
#include "./nlohmann/json.hpp"
#include "./detect/face_landmark.h"
#include <sys/stat.h>
// namespace fs = std::experimental::filesystem; // 修改

typedef struct
{
	std::string model_file;
	std::string input_file;
	int image_width;
	int image_height;
	std::string output_node;
	int use_quant;
} Config;

void readConfig(std::string config_file, Config &cfg)
{
	std::ifstream f(config_file.c_str());
	nlohmann::json data = nlohmann::json::parse(f);
	cfg.input_file = data.at("input_file");
	cfg.model_file = data.at("model_file");
	cfg.use_quant = data.at("use_quant");
	cfg.image_width = data.at("image_width");
	cfg.image_height = data.at("image_height");
	cfg.output_node = data.at("output_node");
}

int main(int argc, char **argv)
{
	Config cfg_data;
	std::string config_file = "./config.json";
	printf(" model file  : \n");
	readConfig(config_file, cfg_data);

	std::string model_name = cfg_data.model_file;
	std::string test_dir = cfg_data.input_file;
	int use_quant = cfg_data.use_quant;
	int image_width = cfg_data.image_width;
	int image_height = cfg_data.image_height;
	std::string output_node = cfg_data.output_node;

	printf(" model file  : %s\n", model_name.c_str());
	printf(" input  file  : %s\n", test_dir.c_str());
	printf(" use quant : %d\n", use_quant);
	sleep(5);

	std::string device_name;
	int device = 0;
	if (use_quant == 0)
	{
		printf("use quantization\n");
		device = -1;
		device_name = "CPU";
	}
	else if (use_quant == 1)
	{
		printf("use GPU\n");
		device = 1;
		device_name = "GPU";
	}
	else if (use_quant == 2)
	{
		printf("use DSP\n");
		device = 2;
		device_name = "DSP";
	}
	sleep(5);

	char save_dir[256];
	sprintf(save_dir, "result_%d_%d_%s", image_width, image_height, device_name.c_str());
	// fs::create_directory(save_dir);
	mkdir(save_dir, S_IRWXU);

	std::vector<std::string> output_node_names{output_node};
	// std::shared_ptr<FaceLandMarker> face_detector = std::make_shared<FaceLandMarker>();
	std::shared_ptr<FaceLandMarker> face_detector = std::shared_ptr<FaceLandMarker>(new FaceLandMarker());
	face_detector->Init(image_width, image_height, model_name, 0.5, output_node_names, device, 1);

	std::vector<cv::String> name_List;
	cv::glob(test_dir, name_List);
	std::ofstream outfile;
	outfile.open("pts_result.txt");
	for (int i = 0; i < name_List.size(); i++)
	{
		int index = name_List[i].rfind("/");
		std::string base_name = name_List[i].substr(index + 1, name_List[i].size());
		// printf("base name %s\n", base_name.c_str());
		std::string file_name = name_List[i];
		char save_name[256];
		sprintf(save_name, "%s/%s", save_dir, base_name.c_str());

		index = base_name.rfind("_");
		std::string base_name_int = base_name.substr(index + 1, base_name.size());
		// index = base_name.find(".");
		base_name_int = base_name_int.substr(0, base_name_int.size() - 4);
		cv::Mat image = cv::imread(file_name);
		printf("save_name %s\n", base_name_int.c_str());
		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
		face_detector->Inference(image, save_name);
		face_detector->save_text(outfile, base_name_int);
	}
	outfile.close();
	return 0;
}
