/***
 *@Author       : xyoung
 *@Date         : 2023-10-19 11:06:20
 *@LastEditTime : 2023-10-19 11:21:16
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <unistd.h>
#include "snpe_infer/snpe_infer.h"
#include "nlohmann/json.hpp"

typedef struct
{
	std::string model_file;
	std::string input_file;
	int image_width;
	int image_height;
	std::string output_node;
	bool use_quant;
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

	// SnpeInfer *detector = new SnpeInfer(model_name, 448, 448, {"outPutNode"}, 0 );
	//  std::unique_ptr<SnpeInfer> detector( new SnpeInfer(model_name, 448,448, {"outPutNode"}));
	// std::unique_ptr<SnpeInfer> detector = std::make_unique<SnpeInfer>(model_name, 448, 448, {"outPutNode"},0);
	// detector.reset(model_name, 448, 448, {"outPutNode"},0);

	int device = 0;
	if (use_quant)
	{
		printf("use quantization\n");
		device = 2;
	}
	else
	{
		printf("use none quantization\n");
		device = 0;
	}
	sleep(5);

	std::unique_ptr<SnpeInfer> detector;
	detector.reset(new SnpeInfer(model_name, image_width, image_height, {output_node}, device));

	std::vector<cv::String> name_List;
	cv::glob(test_dir, name_List);

	float total_time = 0.0;
	for (int i = 0; i < name_List.size(); i++)
	{
		std::string file = name_List.at(i);

		std::cout << "processing file name : " << file << std::endl;
		cv::Mat img = cv::imread(file);
		if (img.empty())
		{
			printf("empty data file %s", file.c_str());
		}

		detector->DoInference(img, 1, true);
	}

	return 0;
}