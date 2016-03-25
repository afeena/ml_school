#include <string>
#include <unordered_map>
#include <vector>
class IO {
public:
	IO();
	IO(const IO& orig);
	virtual ~IO();
	std::unordered_map<int, int> read_train_csv(std::string filename);
	std::vector<int> read_test_csv(std::string filename);
	std::vector<float> feature_indices(std::string filename);
private:

};


