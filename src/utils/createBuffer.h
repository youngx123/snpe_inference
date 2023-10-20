
#include <vector>
#include <string>
#include <cassert>

#include <unordered_map>
#include <iostream>
#include "SNPE/SNPE.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlSystem/UserBufferMap.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorShape.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlSystem/IUserBufferFactory.hpp"
#include "DlSystem/UserBufferMap.hpp"

typedef unsigned int GLuint;

size_t calcSizeFromDims(const zdl::DlSystem::Dimension *dims,
						size_t rank, size_t elementSize);

void createUserBuffer(
	zdl::DlSystem::UserBufferMap &userBufferMap,
	std::unordered_map<std::string, std::vector<uint8_t>> &applicationBuffers,
	std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
	std::unique_ptr<zdl::SNPE::SNPE> &snpe,
	const char *name,
	const bool isTfNBuffer,
	int bitWidth);

// Create a UserBufferMap of the SNPE network inputs
void createInputBufferMap(
	zdl::DlSystem::UserBufferMap &inputMap,
	std::unordered_map<std::string, std::vector<uint8_t>> &applicationBuffers,
	std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
	std::unique_ptr<zdl::SNPE::SNPE> &snpe,
	const bool isTfNBuffer,
	int bitWidth);

// Create a UserBufferMap of the SNPE network outputs
void createOutputBufferMap(
	zdl::DlSystem::UserBufferMap &outputMap,
	std::unordered_map<std::string, std::vector<uint8_t>> &applicationBuffers,
	std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
	std::unique_ptr<zdl::SNPE::SNPE> &snpe,
	const bool isTfNBuffer,
	int bitWidth);

void createUserBuffer(
	zdl::DlSystem::UserBufferMap &userBufferMap,
	std::unordered_map<std::string, GLuint> &applicationBuffers,
	std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
	std::unique_ptr<zdl::SNPE::SNPE> &snpe, const char *name);

void createInputBufferMap(
	zdl::DlSystem::UserBufferMap &inputMap,
	std::unordered_map<std::string, GLuint> &applicationBuffers,
	std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
	std::unique_ptr<zdl::SNPE::SNPE> &snpe);