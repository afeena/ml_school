#include "IO.h"
#include "DB.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <xgboost/base.h>
#include <xgboost/c_api.h>
#include <unordered_map>
#include <vector>
#include <fstream>

std::vector<float> extract_features(std::vector<int> data, std::string dir) {
	cv::Mat img, img_hsv, img_gray;
	std::vector<float> feature_vector;
	for (auto it = data.begin(); it != data.end(); ++it) {
		std::string path = dir + "/" + std::to_string((*it)) + ".jpg";
		img = cv::imread(path);

		cv::cvtColor(img, img_hsv, CV_BGR2HLS);

		int hbins = 10, sbins = 10, lbins = 10;
		int histSize[] = {hbins, sbins, lbins};
		float hranges[] = {0, 256};
		float sranges[] = {0, 256};
		float lranges[] = {0, 256};
		const float* ranges[] = {hranges, sranges, lranges};
		cv::Mat_<float> hist;

		int channels[] = {0, 1, 2};

		cv::calcHist(&img_hsv, 1, channels, cv::Mat(),
			hist, 3, histSize, ranges,
			true, false);

		std::vector<float> hog_descriptors;
		cv::cvtColor(img, img_gray, CV_BGR2GRAY);
		cv::resize(img_gray, img_gray, cv::Size(128, 128), 0, 0);

		cv::Size winSize = cv::Size(128, 128);
		cv::Size blockSize = cv::Size(16, 16);
		cv::Size blockStride = cv::Size(8, 8);
		cv::Size cellSize = cv::Size(8, 8);
		int nbins = 9;
		int derivAperture = 1;
		double winSigma = 4.;
		int histogramNormType = 0;
		double L2HysThreshold = 2.0000000000000001e-01;
		bool gammaCorrection = 0;
		int nlevels = 64;
		cv::HOGDescriptor hog_desc = cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels);
		hog_desc.compute(img_gray, hog_descriptors);

		feature_vector.insert(feature_vector.end(), hist.begin(), hist.end());
		feature_vector.insert(feature_vector.end(), hog_descriptors.begin(), hog_descriptors.end());


	}
	return feature_vector;
}

std::vector<int> convert_prob_into_labels(const float* raw, bst_ulong len) {

	std::vector<int> labels;
	labels.reserve(len);
	for (bst_ulong i = 0; i < len; i++) {
		int label = raw[i] > 0.1 ? 1 : 0;
		labels.push_back(label);
	}
	return labels;

}

int make_predict() {
	IO io;
	std::string dir = "/data/images";
	auto test_data = io.read_test_csv("final_test.csv");
	auto features = extract_features(test_data, dir);
	auto indices = io.feature_indices("indices");
	std::vector<float> important_features;

	bst_ulong num_samples = test_data.size();
	int f_it = 0, i_it = 0;
	int offset_step = features.size() / num_samples;
	int offset = 0;
	do {
		int imp_ind = indices[i_it] + offset;
		important_features.push_back(features[imp_ind]);
		i_it++;
		f_it++;
		if (i_it == 1500) {
			offset += offset_step;
			f_it = offset;
			i_it = 0;
		}

	} while (f_it != features.size());

	cv::Mat_<float> test_set(important_features);
	std::vector<int> predict_res;


	bst_ulong feature_vec_size = (important_features.size()) / num_samples;
	BoosterHandle bst_handle;
	DMatrixHandle xgb_mats[1];
	XGDMatrixCreateFromMat((float*) test_set.data, num_samples, feature_vec_size, 0, &(xgb_mats[0]));
	XGBoosterCreate(xgb_mats, 1, &bst_handle);

	XGBoosterLoadModel(bst_handle, "./booster");

	bst_ulong len;
	const float* out;
	XGBoosterPredict(bst_handle, xgb_mats[0], 0, 0, &len, &out);

	predict_res = convert_prob_into_labels(out, len);
	std::cout << "DONE" << std::endl;

	DB db;
	db.save(test_data, predict_res);

}

int main(int argc, char** argv) {
	make_predict();
	return 0;
}

