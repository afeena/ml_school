#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <map>


using namespace std;

/*
 * 
 */
float cosine_similarity(std::vector<float> &a, std::vector<float> &b) {
	if (a.size() != b.size())
		return -1;

	float mul_sum = 0;
	float sqr_sum_a = 0, sqr_sum_b = 0;
	auto it_a = a.begin();
	auto it_b = b.begin();
	while (it_a != a.end() && it_b != b.end()) {
		mul_sum += (*it_a)*(*it_b);
		sqr_sum_a += pow((*it_a), 2);
		sqr_sum_b += pow((*it_b), 2);
		it_a++;
		it_b++;
	}
	float denominator = (sqrt(sqr_sum_a) * sqrt(sqr_sum_b));
	if(denominator==0)
		return 0;
	float similarity = mul_sum / denominator;
	return similarity;
}

std::unordered_map<std::string, std::vector<float>> read_features(std::string filename) {
	std::string line;
	std::unordered_map<std::string, std::vector<float>> features;
	ifstream feature_file(filename);
	if (feature_file.is_open()) {
		while (!feature_file.eof()) {
			getline(feature_file, line);
			std::string word;
			float w;
			std::stringstream ss(line);
			ss>>word;
			std::vector<float> weights;
			while (!ss.eof()) {
				ss>>w;
				weights.push_back(w);
			}
			features.insert({word, weights});
		}
		feature_file.close();

	}
	return features;
}

std::multimap<float, std::string,std::greater<float>> words_similarity(std::string word, std::unordered_map<std::string, std::vector<float>> &features) {
	std::vector<float> word_features;
	std::multimap<float, std::string,std::greater<float>> result;
	std::unordered_map < std::string, std::vector<float>>::const_iterator got = features.find(word);

	if (got == features.end())
		std::cout << "word not found";
	else
		word_features = got->second;

	for (auto pair : features) {
		if (pair.first == word)
			continue;
		float sim = cosine_similarity(word_features, pair.second);
		result.insert({sim, pair.first});
	}

	return result;


}

int main(int argc, char** argv) {

	if (argc != 2) {
		cout << "Please, enter the word" << std::endl;
		return 0;
	}

	auto features = read_features("/home/mlstudent1/features");
	auto result = words_similarity(argv[1], features);

	int topn = 10;
	int count = 0;
	auto it = result.begin();
	while (count != topn && it != result.end()) {
		std::cout << it->second << " " << it->first << std::endl;
		count++;
		it++;
	}


}

